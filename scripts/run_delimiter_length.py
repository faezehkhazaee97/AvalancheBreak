#!/usr/bin/env python3
"""Delimiter type x longer sequences on dependency tasks (reviewer questions: do the
long-sequence gain and the post-marker blind spot persist beyond N = 64, and do other common
delimiters behave like "|"?).

Tasks: running_xor and running_sum_local (modulus 4), digits, where the blind spot appears; and
random_permutation_digits (alphabet size 8), a position-local control with no blind spot, which
isolates the long-sequence gain. Lengths N in {64, 128, 256}.
Markers: "|" (the original), "," and a line break, each every k in {4, 16} characters, plus
the no-marker baseline. Seeds are those of the main grid (tasks.common.runner._local_seed),
so the "|" cells at N = 64 replicate existing runs. Decoding is greedy, as everywhere.

The prompt is the task's original prompt with (1) "|" in the input replaced by the marker and
(2) the constraint sentence rewritten for that marker. The baseline prompt is the original
no-marker prompt, unchanged.

Scoring follows tasks/common/scoring.py (index by index, prediction clipped to the truth's
length), except that the answer is read across lines for the line-break marker: consecutive
lines made only of digits and spaces are joined. Marker characters and spaces are removed
from the prediction before comparing. A self-check at start verifies that every ground truth,
formatted with each marker, scores 1.0.

Output: results/delimiter_length/<model_tag>.csv (one row per generation, appended; resumable)
Usage:
  python scripts/run_delimiter_length.py --model Qwen/Qwen2.5-32B-Instruct [--dry]
"""

import argparse
import csv
import importlib
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.runner import _local_seed, max_new_tokens  # noqa: E402

ORIGINAL = ("Constraint: The input contains vertical bars (|). These are structural boundaries.\n"
            "DO NOT transform them. Copy them exactly to the output.")
MARKERS = {
    "bar": ("|", ORIGINAL),
    "comma": (",", "Constraint: The input contains commas (,). These are structural boundaries.\n"
                   "DO NOT transform them. Copy them exactly to the output."),
    "newline": ("\n", "Constraint: The input is split across several lines. These line breaks are structural "
                      "boundaries.\nDO NOT transform them. Reproduce the same line breaks in the output."),
}
TASKS = {"running_xor": None, "running_sum_local": 4, "random_permutation_digits": 8}
LENGTHS, KS = [64, 128, 256], [4, 16]


def build(engine, axis, N, k, t, marker):
    rng = random.Random(_local_seed(N, k, axis, t))
    inp, truth, prompt = (engine.build_prompt(N, k, rng, axis_val=axis) if axis
                          else engine.build_prompt(N, k, rng))
    if k == 0:
        return inp, truth, prompt
    sym, constraint = MARKERS[marker]
    head, sep, tail = prompt.rpartition("Input: ")
    assert sep and head.count(ORIGINAL) == 1 and tail.endswith("\nOutput: ")
    body = tail[: -len("\nOutput: ")]
    assert body == inp
    new_inp = inp.replace("|", sym)
    prompt = head.replace(ORIGINAL, constraint) + "Input: " + new_inp + "\nOutput: "
    return new_inp, truth.replace("|", sym), prompt


def extract(raw, marker):
    pred = raw.strip().split("Explanation")[0]
    if marker == "newline":
        lines = []
        for ln in pred.split("\n"):
            s = ln.strip()
            if not s:
                continue
            if re.fullmatch(r"[0-9 ]+", s):
                lines.append(s)
            else:
                break
        pred = "".join(lines)
    else:
        pred = pred.split("\n")[0]
    for ch in ("|", ",", " "):
        pred = pred.replace(ch, "")
    return pred


def score(raw, truth, marker):
    clean_truth = re.sub(r"[|,\n]", "", truth)
    pred = extract(raw, marker)[: len(clean_truth)]
    hits = [i < len(pred) and pred[i] == c for i, c in enumerate(clean_truth)]
    return pred, clean_truth, sum(hits) / len(clean_truth), float(all(hits) and len(pred) == len(clean_truth))


def self_check(engines):
    for task, (engine, axis) in engines.items():
        for N in LENGTHS:
            for k in KS:
                for m in MARKERS:
                    inp, truth, prompt = build(engine, axis, N, k, 0, m)
                    _, ct, tar, sar = score(truth, truth, m)
                    assert tar == 1.0 and sar == 1.0 and len(ct) == N, (task, N, k, m)
                    assert inp.count(MARKERS[m][0]) == (N - 1) // k, (task, N, k, m)
    print("Self-check passed: every truth scores 1.0 under each marker.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true", help="self-check and print example prompts only")
    args = ap.parse_args()

    engines = {t: (importlib.import_module(f"tasks.{t}.engine"), a) for t, a in TASKS.items()}
    self_check(engines)
    if args.dry:
        e, a = engines["running_xor"]
        for m in MARKERS:
            print(f"----- {m} -----\n{build(e, a, 64, 16, 0, m)[2]}")
        print("----- base -----\n" + build(e, a, 64, 0, 0, "bar")[2])
        return

    import torch
    from transformers import AutoTokenizer
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "delimiter_length"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Marker", "Interval", "Seq_Length", "Axis", "Trial", "TAR", "SAR",
              "Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Marker"], r["Interval"], r["Seq_Length"], r["Trial"]) for r in csv.DictReader(f)}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()

    plans = [("base", 0)] + [(m, k) for m in MARKERS for k in KS]
    for N in LENGTHS:
        for task, (engine, axis) in engines.items():
            for marker, k in plans:
                for t in range(args.trials):
                    key = (task, marker, str(k), str(N), str(t))
                    if key in done:
                        continue
                    inp, truth, prompt = build(engine, axis, N, k, t, marker)
                    enc = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                       add_generation_prompt=True, return_tensors="pt",
                                                       return_dict=True, enable_thinking=False)
                    enc = {kk: v.to(model.device) for kk, v in enc.items()}
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=max_new_tokens(engine, N, k), do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)
                    raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    pred, ct, tar, sar = score(raw, truth, marker if k else "bar")
                    new = not out_csv.exists()
                    with open(out_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        if new:
                            w.writeheader()
                        w.writerow(dict(Task=task, Marker=marker, Interval=k, Seq_Length=N, Axis=axis or 2,
                                        Trial=t, TAR=tar, SAR=sar, Input_Given=inp, Ground_Truth=ct,
                                        LLM_Output=pred, Raw_Model_Output=raw))
                print(f"done N={N} {task} {marker} k={k}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
