#!/usr/bin/env python3
"""Paired-input replication of the main grid (reviewer question).

In the main grids the no-marker and marker conditions use different random inputs (the input seed
depends on the spacing k), so every reported delta is an unpaired comparison. Here the input of
trial t at length N is drawn from the k = 0 seed whatever the spacing, so the no-marker run and
every marker run of a trial transform the same string, and the delta can be estimated as a mean of
paired differences. The no-marker rows reuse the main grid's seeds exactly, so with greedy decoding
they should reproduce the main-grid Base rows (checked by scripts/analyze_paired_inputs.py).

Tasks: the two tokenization controls for the position-local tasks (spaced letters, digits) and the
four dependency tasks. N in {16, 32, 64}, k in {0, 1, 2, 4, 8, 16} (k < N), every alphabet or
modulus value, 20 trials, greedy.

Output: results/paired_inputs/<model_tag>.csv
Usage:  python scripts/run_paired_inputs.py --model Qwen/Qwen2.5-7B-Instruct [--dry]
"""

import argparse
import csv
import importlib
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common import scoring  # noqa: E402
from tasks.common.boundary import valid_intervals  # noqa: E402
from tasks.common.runner import _local_seed, max_new_tokens  # noqa: E402

TASKS = ["cyclic_shift_spaced", "random_permutation_spaced", "conditional_shift_spaced",
         "cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits",
         "copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]
LENGTHS = [16, 32, 64]
KS = [0, 1, 2, 4, 8, 16]


def build(engine, axis, N, k, t):
    """Same input for every k: the rng is seeded as the k = 0 cell of the same trial."""
    rng = random.Random(_local_seed(N, 0, axis, t))
    return engine.build_prompt(N, k, rng, axis_val=axis) if engine.AXIS_NAME else engine.build_prompt(N, k, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    engines = {t: importlib.import_module(f"tasks.{t}.engine") for t in TASKS}
    total = 0
    for t, e in engines.items():
        for axis in (e.AXIS_VALUES or [None]):
            for N in LENGTHS:
                ks = valid_intervals(N, KS)
                plain = {k: build(e, axis, N, k, 0)[0].replace("|", "").replace(" ", "") for k in ks}
                assert len(set(plain.values())) == 1, f"{t}: inputs differ across k"
                truth0 = build(e, axis, N, 0, 0)[1]
                assert scoring.score(truth0, truth0)["TAR"] == 1.0
                total += len(ks) * args.trials
    print(f"Self-check passed: within a trial the input is identical at every spacing. {total} generations per model.")
    if args.dry:
        return

    import torch
    from transformers import AutoTokenizer
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "paired_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Method", "Interval", "Seq_Length", "Axis", "Trial", "TAR", "SAR",
              "Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Interval"], r["Seq_Length"], r["Axis"], r["Trial"]) for r in csv.DictReader(f)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()
    for task in TASKS:
        engine = engines[task]
        for axis in (engine.AXIS_VALUES or [None]):
            for N in LENGTHS:
                for k in valid_intervals(N, KS):
                    for t in range(args.trials):
                        if (task, str(k), str(N), str(axis or 2), str(t)) in done:
                            continue
                        inp, truth, prompt = build(engine, axis, N, k, t)
                        enc = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                           add_generation_prompt=True, return_tensors="pt",
                                                           return_dict=True, enable_thinking=False)
                        enc = {kk: v.to(model.device) for kk, v in enc.items()}
                        with torch.no_grad():
                            out = model.generate(**enc, max_new_tokens=max_new_tokens(engine, N, k), do_sample=False,
                                                 pad_token_id=tokenizer.eos_token_id)
                        raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                        res = scoring.score(raw, truth)
                        new = not out_csv.exists()
                        with open(out_csv, "a", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=fields)
                            if new:
                                w.writeheader()
                            w.writerow(dict(Task=task, Method="Base" if k == 0 else "Cellular_Bounds", Interval=k,
                                            Seq_Length=N, Axis=axis or 2, Trial=t, TAR=res["TAR"], SAR=res["SAR"],
                                            Input_Given=inp, Ground_Truth=res["clean_truth"],
                                            LLM_Output=res["clean_pred"], Raw_Model_Output=raw))
                    print(f"done {task} axis={axis} N={N} k={k}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
