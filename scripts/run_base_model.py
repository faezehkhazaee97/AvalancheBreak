#!/usr/bin/env python3
"""Base (pretrained) checkpoint versus its instruction-tuned version (reviewer question).

Every model in the paper is instruction-tuned. A base checkpoint cannot be given the chat-formatted
instruction, so both checkpoints are prompted the same new way here: the task's own instruction
text, no chat template, followed by three solved examples in the Input/Output format of the query
(same length and spacing as the query, drawn from a separate seed stream) and then the query. The
model completes the "Output:" line. The only difference between the two runs is the weights.

Tasks: the four dependency tasks and the three digit controls. N in {16, 32, 64}, k in
{0, 2, 4, 8, 16} (k < N), every alphabet or modulus value, 20 trials, greedy.

Output: results/base_model/<model_tag>.csv
Usage:  python scripts/run_base_model.py --model Qwen/Qwen2.5-7B [--dry]
        python scripts/run_base_model.py --model Qwen/Qwen2.5-7B-Instruct
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

TASKS = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max",
         "cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"]
LENGTHS = [16, 32, 64]
KS = [0, 2, 4, 8, 16]
SHOTS = 3


def gen(engine, axis, N, k, rng):
    return engine.build_prompt(N, k, rng, axis_val=axis) if engine.AXIS_NAME else engine.build_prompt(N, k, rng)


def build(engine, axis, N, k, t):
    """The grid prompt of (N, k, axis, t) with three solved examples inserted before the query."""
    inp, truth, prompt = gen(engine, axis, N, k, random.Random(_local_seed(N, k, axis, t)))
    head, sep, tail = prompt.rpartition("Input: ")
    assert sep and tail == f"{inp}\nOutput: ", "unexpected prompt tail"
    shots, j = [], 0
    while len(shots) < SHOTS:
        s_in, s_out, _ = gen(engine, axis, N, k, random.Random(f"shots-{engine.TASK_NAME}-{N}-{k}-{axis}-{t}-{j}"))
        j += 1
        if s_in != inp:
            shots.append(f"Input: {s_in}\nOutput: {s_out}\n\n")
    return inp, truth, head + "".join(shots) + f"Input: {inp}\nOutput:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    engines = {t: importlib.import_module(f"tasks.{t}.engine") for t in TASKS}
    total = 0
    for t, e in engines.items():
        for axis in (e.AXIS_VALUES or [None]):
            for N in LENGTHS:
                for k in valid_intervals(N, KS):
                    inp, truth, p = build(e, axis, N, k, 0)
                    assert p.count("Input: ") == SHOTS + 1 and p.endswith(f"Input: {inp}\nOutput:")
                    assert scoring.score(truth, truth)["TAR"] == 1.0
                    total += args.trials
    print(f"Self-check passed: every prompt carries {SHOTS} solved examples and ends at the query. "
          f"{total} generations per model.")
    if args.dry:
        print(build(engines["running_xor"], None, 16, 4, 0)[2])
        return

    import torch
    from transformers import AutoTokenizer
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "base_model"
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
                        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                        enc = {kk: v.to(model.device) for kk, v in enc.items()}
                        with torch.no_grad():
                            out = model.generate(**enc, max_new_tokens=max_new_tokens(engine, N, k), do_sample=False,
                                                 pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id)
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
