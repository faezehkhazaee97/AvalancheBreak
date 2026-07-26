"""Paid-API pilot runner: 2 trials per task (one Base, one Cellular_Bounds),
using the EXACT SAME prompt-construction logic and seed formula
(tasks.common.runner._local_seed) as the real HuggingFace-model grid runs,
so these are genuinely "the same examples we used," not freshly sampled
ones -- just re-derived locally (build_prompt only depends on N/k/axis/rng,
never on which model will consume it, so this is model-agnostic and free).

Two modes:
  --dry-run   : construct and print/save every prompt, make ZERO API calls.
                Always run this first and review the output before spending
                money.
  (default)   : actually calls the Gemini API for each prompt and saves
                everything -- full prompt, raw response text, parsed
                prediction, ground truth, TAR/SAR, and whatever usage/token
                metadata the API returns, to a CSV, so nothing is lost and
                nothing needs to be re-run to recover missing information.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment for anything
other than --dry-run.
"""

import argparse
import csv
import importlib
import os
import random
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.common.runner import _local_seed  # noqa: E402
from tasks.common import scoring as scoring_mod  # noqa: E402

# One (N, k) pair per task: (Base cell, Cellular_Bounds cell). N=16 across
# the board -- comfortably mid-range for every task's own LENGTHS list, and
# k=4 is a valid boundary density for every task at N=16 (checked against
# each task's own valid_intervals). axis_val picks the middle value where a
# task has an AXIS_VALUES sweep, matching the convention used elsewhere in
# this study for a single representative axis point.
PILOT_TASKS = {
    "cyclic_shift":            {"axis_val": 4},
    "character_substitution":  {"axis_val": 4},
    "random_permutation":      {"axis_val": 4},
    "conditional_shift":       {"axis_val": 4},
    "composed_permutation":    {"axis_val": 4},
    "running_xor":             {"axis_val": None},
    "running_sum_local":       {"axis_val": 4},
    "copy_previous":           {"axis_val": None},
    "running_max":             {"axis_val": 4},
}
N = 16
K_BOUNDS = 4
TRIAL = 0


def build_pilot_examples():
    """Returns a list of dicts, one per (task, condition), with the fully
    constructed prompt and ground truth -- no API calls."""
    examples = []
    for task_name, cfg in PILOT_TASKS.items():
        engine = importlib.import_module(f"tasks.{task_name}.engine")
        axis_val = cfg["axis_val"]
        for method, k in [("Base", 0), ("Cellular_Bounds", K_BOUNDS)]:
            seed = _local_seed(N, k, axis_val, TRIAL)
            rng = random.Random(seed)
            if engine.AXIS_NAME:
                input_str, truth, prompt = engine.build_prompt(N, k, rng, axis_val=axis_val)
            else:
                input_str, truth, prompt = engine.build_prompt(N, k, rng)
            examples.append({
                "Task": task_name, "Method": method, "Interval": k, "Seq_Length": N,
                "Axis_Val": axis_val, "Trial": TRIAL,
                "Input_Given": input_str, "Ground_Truth": truth, "Prompt": prompt,
                "Scoring_Mode": getattr(engine, "SCORING_MODE", "char"),
                "Token_Regex": getattr(engine, "TOKEN_REGEX", None),
            })
    return examples


def call_gemini(prompt, model_name, client, max_output_tokens=200):
    """Returns (raw_text, usage_dict, error_str_or_None).

    Explicitly disables "thinking" (thinking_budget=0) and sets
    temperature=0 -- these tasks are simple deterministic transformations
    with no need for extended reasoning, thinking tokens are billed as
    output tokens but invisible in resp.text (would silently inflate cost
    beyond what a naive prompt+output token estimate suggests), and
    temperature=0 matches the greedy decoding (do_sample=False) used for
    every other model in this study, for methodological consistency.
    max_output_tokens caps worst-case cost per call if something goes wrong.
    """
    from google.genai import types
    config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    try:
        resp = client.models.generate_content(model=model_name, contents=prompt, config=config)
        raw_text = resp.text if resp.text is not None else ""
        usage = {}
        if getattr(resp, "usage_metadata", None) is not None:
            um = resp.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", None),
                "candidates_tokens": getattr(um, "candidates_token_count", None),
                "total_tokens": getattr(um, "total_token_count", None),
            }
        # model_name may be an alias (e.g. "gemini-flash-latest") that
        # silently resolves to a different concrete model over time --
        # resp.model_version is the actual resolved model, recorded
        # separately so a run stays attributable after the fact even if
        # the alias points somewhere else next time this is run.
        resolved_model = getattr(resp, "model_version", None)
        return raw_text, usage, resolved_model, None
    except Exception as e:
        return "", {}, None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Construct and print prompts, make zero API calls.")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to results_clean/gemini_pilot/<model>.csv")
    args = parser.parse_args()

    examples = build_pilot_examples()

    if args.dry_run:
        print(f"Constructed {len(examples)} examples ({len(PILOT_TASKS)} tasks x 2 conditions each). Zero API calls made.\n")
        for ex in examples:
            print("=" * 80)
            print(f"Task: {ex['Task']} | Method: {ex['Method']} | N={ex['Seq_Length']} | k={ex['Interval']} | axis={ex['Axis_Val']}")
            print(f"Scoring mode: {ex['Scoring_Mode']}" + (f" (token_regex={ex['Token_Regex']})" if ex['Token_Regex'] else ""))
            print("-" * 80)
            print(ex["Prompt"])
            print(f"[Ground truth: {ex['Ground_Truth']}]")
        print("=" * 80)
        print(f"\nTotal: {len(examples)} prompts, 0 API calls, $0 spent.")
        return

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY or GOOGLE_API_KEY before running without --dry-run.")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    output_path = args.output or os.path.join(REPO_ROOT, "results_clean", "gemini_pilot", f"{args.model}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["Task", "Method", "Interval", "Seq_Length", "Axis_Val", "Trial",
                  "Model_Requested", "Model_Resolved", "Timestamp_UTC", "Input_Given", "Ground_Truth", "Prompt",
                  "Raw_Response", "LLM_Output", "TAR", "SAR", "First_Error_Pos",
                  "Prompt_Tokens", "Candidates_Tokens", "Total_Tokens", "Error"]

    rows = []
    for i, ex in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] {ex['Task']} / {ex['Method']} ...")
        raw_text, usage, resolved_model, error = call_gemini(ex["Prompt"], args.model, client)

        if error:
            result = {"clean_pred": "", "TAR": None, "SAR": None, "First_Error_Pos": None}
        elif ex["Scoring_Mode"] == "token":
            result = scoring_mod.score_tokens(raw_text, ex["Ground_Truth"], token_regex=ex["Token_Regex"])
        else:
            result = scoring_mod.score(raw_text, ex["Ground_Truth"])

        rows.append({
            "Task": ex["Task"], "Method": ex["Method"], "Interval": ex["Interval"],
            "Seq_Length": ex["Seq_Length"], "Axis_Val": ex["Axis_Val"], "Trial": ex["Trial"],
            "Model_Requested": args.model, "Model_Resolved": resolved_model,
            "Timestamp_UTC": datetime.now(timezone.utc).isoformat(),
            "Input_Given": ex["Input_Given"], "Ground_Truth": ex["Ground_Truth"], "Prompt": ex["Prompt"],
            "Raw_Response": raw_text, "LLM_Output": result.get("clean_pred", ""),
            "TAR": result["TAR"], "SAR": result["SAR"], "First_Error_Pos": result.get("First_Error_Pos"),
            "Prompt_Tokens": usage.get("prompt_tokens"), "Candidates_Tokens": usage.get("candidates_tokens"),
            "Total_Tokens": usage.get("total_tokens"), "Error": error or "",
        })
        # write incrementally after every call -- never lose a paid call to a crash
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        time.sleep(1)

    print(f"\nSaved {len(rows)} rows to {output_path}")
    n_errors = sum(1 for r in rows if r["Error"])
    if n_errors:
        print(f"WARNING: {n_errors} call(s) errored -- see Error column.")


if __name__ == "__main__":
    main()
