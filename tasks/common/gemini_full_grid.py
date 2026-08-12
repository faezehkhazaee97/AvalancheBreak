"""Full-grid Gemini API run: every valid (N, k, axis) cell across all 9
core tasks, 1 trial per cell (Trial=0), reusing the exact same grid-building
logic (tasks.common.boundary.valid_intervals) and local-seed formula
(tasks.common.runner._local_seed) as the real HuggingFace-model grid runs --
same cells, same prompts, just a different model behind the API.

Reuses call_gemini() from gemini_runner.py unchanged (thinking disabled,
temperature=0, max_output_tokens cap, model_version capture, per-call error
handling) so the exact same cost/consistency safeguards apply here.

Writes incrementally after every call -- a crash partway through never loses
already-completed (already-paid-for) rows; rerunning just needs the caller to
note where output_path already has data if resuming is wanted (not built in,
this run is small enough that a clean restart is cheap).
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
from tasks.common.boundary import valid_intervals, STANDARD_LENGTHS, STANDARD_INTERVALS  # noqa: E402
from tasks.common import scoring as scoring_mod  # noqa: E402
from tasks.common.gemini_runner import call_gemini  # noqa: E402

TASKS = ["cyclic_shift", "character_substitution", "random_permutation", "conditional_shift",
         "composed_permutation", "running_xor", "running_sum_local", "copy_previous", "running_max"]

# External-benchmark tasks (CLRS-Text minimum/kadane, and modsum as the
# parity@modulus=2 check) -- kept as a separate list rather than folded into
# TASKS so a bare re-run of this script (no --tasks) reproduces exactly the
# original 9-task grid; pass --tasks clrs_minimum,clrs_kadane,modsum to run
# just these.
EXTERNAL_TASKS = ["clrs_minimum", "clrs_kadane", "modsum"]


def _max_output_tokens_for(engine, N):
    """Mirrors tasks.common.runner._run_one's max_new_tokens fallback chain,
    plus a small safety margin. Deliberately NOT padded for "thinking"
    tokens: the "gemini-flash-latest" alias resolves (as of this run) to
    gemini-3.6-flash, which cannot fully disable thinking -- confirmed by
    direct testing (thinking_budget=0 is rejected outright; explicit 1/100/
    500/-1 all still landed at 1,900-3,700+ actual thought tokens on the
    hardest real cell in this grid, clrs_kadane N=128, well past any
    reasonable fixed buffer and enough to truncate the real answer even at
    max_output_tokens=8000). The fix is not a bigger buffer, it's pinning
    the model: "gemini-3.5-flash" (this study's original 9-task grid) is
    still directly callable (not deprecated) and honors thinking_budget=0
    cleanly -- verified against the same N=128 cells with zero thought-token
    overhead and complete, untruncated answers using exactly this estimate,
    no padding. See DEFAULT_MODEL below."""
    if hasattr(engine, "estimate_max_new_tokens"):
        base = engine.estimate_max_new_tokens(N)
    else:
        base = int(N * 1.5) + 20 if engine.OUTPUT_HAS_BOUNDARIES else N + 20
    return base + 50


def build_full_grid_examples(trial, task_list=None):
    examples = []
    for task_name in (task_list or TASKS):
        engine = importlib.import_module(f"tasks.{task_name}.engine")
        axis_values = engine.AXIS_VALUES or [None]
        lengths = getattr(engine, "LENGTHS", STANDARD_LENGTHS)
        intervals = getattr(engine, "INTERVALS", STANDARD_INTERVALS)
        for N in lengths:
            for k in valid_intervals(N, intervals):
                for axis_val in axis_values:
                    method = "Base" if k == 0 else "Cellular_Bounds"
                    seed = _local_seed(N, k, axis_val, trial)
                    rng = random.Random(seed)
                    if engine.AXIS_NAME:
                        input_str, truth, prompt = engine.build_prompt(N, k, rng, axis_val=axis_val)
                    else:
                        input_str, truth, prompt = engine.build_prompt(N, k, rng)
                    examples.append({
                        "Task": task_name, "Method": method, "Interval": k, "Seq_Length": N,
                        "Axis_Val": axis_val, "Trial": trial,
                        "Input_Given": input_str, "Ground_Truth": truth, "Prompt": prompt,
                        "Scoring_Mode": getattr(engine, "SCORING_MODE", "char"),
                        "Token_Regex": getattr(engine, "TOKEN_REGEX", None),
                        "Max_Output_Tokens": _max_output_tokens_for(engine, N),
                    })
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="gemini-3.5-flash", help="Pinned, not the \"-latest\" alias: the alias silently drifted from gemini-3.5-flash (this study's original grid) to gemini-3.6-flash mid-project, and the newer model can't fully disable thinking (see _max_output_tokens_for's docstring). gemini-3.5-flash is still directly callable and matches the original methodology exactly.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--trial", type=int, default=0, help="Trial index -- matches tasks.common.runner's local seed formula, so Trial=1/2 are genuinely independent samples from Trial=0, poolable into a 3-trial-per-cell dataset.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names to override the default 9-task TASKS list, e.g. clrs_minimum,clrs_kadane,modsum.")
    args = parser.parse_args()

    task_list = args.tasks.split(",") if args.tasks else None
    examples = build_full_grid_examples(args.trial, task_list=task_list)

    if args.dry_run:
        print(f"Constructed {len(examples)} examples across {len(task_list or TASKS)} tasks. Zero API calls made.")
        for ex in examples[:3] + examples[-3:]:
            print(f"  {ex['Task']} N={ex['Seq_Length']} k={ex['Interval']} axis={ex['Axis_Val']} trial={ex['Trial']} max_out={ex['Max_Output_Tokens']}")
        return

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY or GOOGLE_API_KEY before running without --dry-run.")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    output_path = args.output or os.path.join(REPO_ROOT, "results_clean", "gemini_pilot", f"{args.model}_full_grid_trial{args.trial}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["Task", "Method", "Interval", "Seq_Length", "Axis_Val", "Trial",
                  "Model_Requested", "Model_Resolved", "Timestamp_UTC", "Input_Given", "Ground_Truth", "Prompt",
                  "Raw_Response", "LLM_Output", "TAR", "SAR", "First_Error_Pos",
                  "Prompt_Tokens", "Candidates_Tokens", "Total_Tokens", "Error"]

    rows = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(examples) - i - 1) / rate if rate > 0 else 0
            print(f"[{i+1}/{len(examples)}] {ex['Task']} N={ex['Seq_Length']} k={ex['Interval']} ... (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        raw_text, usage, resolved_model, error = call_gemini(ex["Prompt"], args.model, client, max_output_tokens=ex["Max_Output_Tokens"])

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
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        time.sleep(0.5)

    total_tokens = sum(r["Total_Tokens"] or 0 for r in rows)
    n_errors = sum(1 for r in rows if r["Error"])
    print(f"\nDone. Saved {len(rows)} rows to {output_path}")
    print(f"Total tokens used: {total_tokens:,}")
    if n_errors:
        print(f"WARNING: {n_errors} call(s) errored -- see Error column.")


if __name__ == "__main__":
    main()
