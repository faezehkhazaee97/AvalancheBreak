"""Runner variant for the external-tasks suite (procbench_substitute,
string_copy, flip_flop_high, flip_flop_low): supports (a) a Base-only
screening pass across lengths, and (b) a full Base+Cellular_Bounds grid --
both writing each row to CSV immediately after it's computed (not
buffered to a DataFrame and written once at the end like common.runner),
and both resumable: on restart, already-completed (N, k, axis_val, trial)
cells are read back from the existing CSV and skipped, so a killed/
requeued job picks up where it left off instead of redoing finished work
or duplicating rows.

Reuses the exact same local-seed formula and model-loading path as
common.runner so results stay reproducible/comparable with the rest of
the study.
"""

import csv
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import scoring as scoring_mod
from .boundary import STANDARD_LENGTHS, STANDARD_INTERVALS, valid_intervals
from .runner import GLOBAL_SEED, _seed_everything, _local_seed


def _load_done_keys(output_csv, key_cols):
    """Returns the set of already-completed grid-cell keys found in an
    existing output CSV, so a resumed run can skip them."""
    done = set()
    if not os.path.exists(output_csv):
        return done
    with open(output_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(row[c] for c in key_cols)
            done.add(key)
    return done


def _append_row(output_csv, row, fieldnames):
    file_exists = os.path.exists(output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _load_model(model_id):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


def _generate_row(engine, tokenizer, model, N, k, axis_val, t):
    method = "Base" if k == 0 else "Cellular_Bounds"
    seed = _local_seed(N, k, axis_val, t)
    rng = random.Random(seed)

    if engine.AXIS_NAME:
        input_str, truth, prompt = engine.build_prompt(N, k, rng, axis_val=axis_val)
    else:
        input_str, truth, prompt = engine.build_prompt(N, k, rng)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}

    if hasattr(engine, "estimate_max_new_tokens"):
        max_new = engine.estimate_max_new_tokens(N)
    else:
        max_new = int(N * 1.5) + 20 if engine.OUTPUT_HAS_BOUNDARIES else N + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)
    result = scoring_mod.score(raw_output, truth)

    row = {
        "Method": method, "Interval": k, "Seq_Length": N,
        "Alphabet_Size": axis_val if axis_val is not None else "",
        "Trial": t, "Input_Given": input_str, "Ground_Truth": result["clean_truth"],
        "LLM_Output": result["clean_pred"], "TAR": result["TAR"], "SAR": result["SAR"],
        "First_Error_Pos": result["First_Error_Pos"], "Raw_Model_Output": raw_output,
    }
    return row


FIELDNAMES = ["Method", "Interval", "Seq_Length", "Alphabet_Size", "Trial",
              "Input_Given", "Ground_Truth", "LLM_Output", "TAR", "SAR",
              "First_Error_Pos", "Raw_Model_Output"]
KEY_COLS = ["Method", "Interval", "Seq_Length", "Alphabet_Size", "Trial"]


def run_screen(engine, model_id, output_csv, trials_per_cell=10):
    """Base-only (k=0) pass across every length in engine.LENGTHS (or
    STANDARD_LENGTHS if unset), for every axis value if the task has one.
    Appends incrementally, resumable."""
    _seed_everything()
    tokenizer, model = _load_model(model_id)

    lengths = getattr(engine, "LENGTHS", STANDARD_LENGTHS)
    axis_values = engine.AXIS_VALUES or [None]

    done = _load_done_keys(output_csv, KEY_COLS)
    print(f"Resuming: {len(done)} cells already done in {output_csv}" if done else "Starting fresh.")

    cells = [(N, 0, axis_val, t) for N in lengths for axis_val in axis_values for t in range(trials_per_cell)]
    n_run = 0
    for N, k, axis_val, t in tqdm(cells, desc=f"{engine.TASK_NAME} (screen)"):
        key = ("Base", str(k), str(N), str(axis_val if axis_val is not None else ""), str(t))
        if key in done:
            continue
        try:
            row = _generate_row(engine, tokenizer, model, N, k, axis_val, t)
            _append_row(output_csv, row, FIELDNAMES)
            n_run += 1
        except Exception as exc:
            print(f"Screen cell failed: N={N} axis={axis_val} t={t} | Error: {exc}")
    print(f"Screen complete: {n_run} new cells run, {len(done)} were already done. Output: {output_csv}")


def run_grid_resumable(engine, model_id, output_csv, trials_per_cell=20):
    """Full Base + Cellular_Bounds grid, same STANDARD_LENGTHS/INTERVALS
    convention as the main study. Appends incrementally, resumable."""
    _seed_everything()
    tokenizer, model = _load_model(model_id)

    lengths = getattr(engine, "LENGTHS", STANDARD_LENGTHS)
    intervals = getattr(engine, "INTERVALS", STANDARD_INTERVALS)
    axis_values = engine.AXIS_VALUES or [None]

    done = _load_done_keys(output_csv, KEY_COLS)
    print(f"Resuming: {len(done)} cells already done in {output_csv}" if done else "Starting fresh.")

    cells = []
    for N in lengths:
        for k in valid_intervals(N, intervals):
            for axis_val in axis_values:
                for t in range(trials_per_cell):
                    cells.append((N, k, axis_val, t))

    n_run = 0
    for N, k, axis_val, t in tqdm(cells, desc=f"{engine.TASK_NAME} (grid)"):
        method = "Base" if k == 0 else "Cellular_Bounds"
        key = (method, str(k), str(N), str(axis_val if axis_val is not None else ""), str(t))
        if key in done:
            continue
        try:
            row = _generate_row(engine, tokenizer, model, N, k, axis_val, t)
            _append_row(output_csv, row, FIELDNAMES)
            n_run += 1
        except Exception as exc:
            print(f"Grid cell failed: N={N} k={k} axis={axis_val} t={t} | Error: {exc}")
    print(f"Grid complete: {n_run} new cells run, {len(done)} were already done. Output: {output_csv}")
