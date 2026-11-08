"""Shared grid-execution harness for all 6 tasks.

Each task's engine.py defines TASK_NAME, AXIS_NAME/AXIS_VALUES (or None),
OUTPUT_HAS_BOUNDARIES, and build_prompt(N, k, rng, axis_val). This module
handles: model loading, seeding, grid iteration (respecting
boundary.valid_intervals to avoid degenerate k>=N cells), generation,
scoring (via common.scoring, which fixes the truncation bug), and CSV
output in either "verify" mode (1 trial per cell) or "grid" mode
(trials_per_cell trials per cell).

Raw_Model_Output (the full, untruncated generation) is now always saved, in
both modes -- it's the one thing that can't be regenerated after the fact,
and it's what two diagnostic metrics (Overlong, Boundary_Fidelity, see
common.metrics) need but couldn't get from the already-clipped/bar-stripped
LLM_Output column alone. Prompt is deliberately NOT stored: every prompt is
100% deterministically reconstructible from (N, k, axis_val, Trial) via the
same local seed used at generation time, so storing it would just be
repeating ~500 bytes of near-identical boilerplate on every single row for
zero information gain. Use reconstruct_prompt() below to regenerate one for
eyeballing.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from . import scoring as scoring_mod
from .boundary import STANDARD_LENGTHS, STANDARD_INTERVALS, valid_intervals

GLOBAL_SEED = 42


def _seed_everything():
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)


def _local_seed(N, k, axis_val, t):
    return 90_000 * N + 100 * k + 10 * (axis_val or 0) + t


def reconstruct_prompt(engine, N, k, axis_val, t):
    """Regenerates the exact prompt used for a given grid cell/trial, since
    it's never stored -- deterministic given (N, k, axis_val, t) via the
    same local seed used at generation time."""
    rng = random.Random(_local_seed(N, k, axis_val, t))
    if engine.AXIS_NAME:
        _, _, prompt = engine.build_prompt(N, k, rng, axis_val=axis_val)
    else:
        _, _, prompt = engine.build_prompt(N, k, rng)
    return prompt


def max_new_tokens(engine, N, k):
    """Generation budget for one cell.

    int(1.5N)+20 covers a marked output at k >= 2 (N + N/2 tokens at most).
    At k = 1 (a marker after every character) the correct output is 2N-1
    tokens for tokenizers that split letters and markers, which exceeds
    int(1.5N)+20 once N > 42: every k=1 output at N >= 64 used to be cut off
    (58 of 64 characters at N = 64). k = 1 therefore gets 2N+20. Decoding is
    greedy, so a larger budget only extends outputs that were cut off; cells
    that never hit the old budget are unchanged."""
    if hasattr(engine, "estimate_max_new_tokens"):
        return engine.estimate_max_new_tokens(N)
    if not engine.OUTPUT_HAS_BOUNDARIES:
        return N + 20
    return 2 * N + 20 if k == 1 else int(N * 1.5) + 20


def load_model(model_id):
    """Text-only models (Qwen2.5, Mistral, Phi) load exactly as before.
    Vision-language checkpoints (e.g. Qwen3.8 / Gemma 4, which carry a
    vision_config) load with their own class: AutoModelForCausalLM can resolve
    them to a text-only class whose weight names don't match the checkpoint.
    Generation here is text-only either way."""
    cfg = AutoConfig.from_pretrained(model_id)
    cls = AutoModelForImageTextToText if getattr(cfg, "vision_config", None) is not None else AutoModelForCausalLM
    return cls.from_pretrained(
        model_id, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )


def _build_grid(engine, trials_per_cell, trial_start=0):
    axis_values = engine.AXIS_VALUES or [None]
    lengths = getattr(engine, "LENGTHS", STANDARD_LENGTHS)
    intervals = getattr(engine, "INTERVALS", STANDARD_INTERVALS)
    tasks = []
    for N in lengths:
        for k in valid_intervals(N, intervals):
            for axis_val in axis_values:
                for t in range(trial_start, trial_start + trials_per_cell):
                    tasks.append((N, k, axis_val, t))
    return tasks


def run(engine, model_id, mode, output_csv, trials_per_cell=None, trial_start=0):
    """mode: 'verify' (1 trial/cell, for a quick pre-run sanity check) or
    'grid' (trials_per_cell trials/cell, production). Both now write the
    same columns, including Raw_Model_Output -- 'verify' vs 'grid' only
    differs in trials_per_cell. trial_start: first trial index (0 for a
    first batch; 40 for a second, independent, non-overlapping batch at
    the study's usual 40-trials-per-batch convention -- same local-seed
    formula, so trial_start=40 with trials_per_cell=40 reuses no seeds
    from a trial_start=0 run)."""
    assert mode in ("verify", "grid")
    if trials_per_cell is None:
        trials_per_cell = 1 if mode == "verify" else 20

    print(f"Task: {engine.TASK_NAME} | Model: {model_id} | Mode: {mode} | Trials/cell: {trials_per_cell}")
    _seed_everything()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id)
    model.eval()

    grid = _build_grid(engine, trials_per_cell, trial_start=trial_start)
    print(f"Grid size: {len(grid)} evaluations")

    rows = []
    for N, k, axis_val, t in tqdm(grid, desc=f"{engine.TASK_NAME} ({mode})"):
        try:
            rows.append(_run_one(engine, tokenizer, model, N, k, axis_val, t))
        except Exception as exc:
            print(f"Task failed: N={N} k={k} axis={axis_val} t={t} | Error: {exc}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")

    summary_cols = ["Method", "Seq_Length"] + ([engine.AXIS_NAME] if engine.AXIS_NAME else [])
    print(df.groupby(summary_cols)[["TAR", "SAR"]].mean().to_string())
    return df


def _run_one(engine, tokenizer, model, N, k, axis_val, t):
    method = "Base" if k == 0 else "Cellular_Bounds"
    seed = _local_seed(N, k, axis_val, t)
    rng = random.Random(seed)

    if engine.AXIS_NAME:
        input_str, truth, prompt = engine.build_prompt(N, k, rng, axis_val=axis_val)
    else:
        input_str, truth, prompt = engine.build_prompt(N, k, rng)

    # enable_thinking=False: hybrid-reasoning models (Qwen3.x, Gemma 4) would
    # otherwise emit a reasoning block before the answer. Templates that don't
    # use the variable ignore it (Qwen2.5's rendered prompt is byte-identical).
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}

    max_new = max_new_tokens(engine, N, k)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)

    if getattr(engine, "SCORING_MODE", "char") == "token":
        result = scoring_mod.score_tokens(raw_output, truth, token_regex=getattr(engine, "TOKEN_REGEX", None))
    else:
        result = scoring_mod.score(raw_output, truth)

    row = {
        "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "Input_Given": input_str, "Ground_Truth": result["clean_truth"],
        "LLM_Output": result["clean_pred"], "TAR": result["TAR"], "SAR": result["SAR"],
        "First_Error_Pos": result["First_Error_Pos"], "Raw_Model_Output": raw_output,
    }
    if engine.AXIS_NAME:
        row[engine.AXIS_NAME] = axis_val
    return row
