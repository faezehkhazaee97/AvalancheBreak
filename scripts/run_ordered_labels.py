#!/usr/bin/env python3
"""Distinct ordered boundary labels vs a repeating marker (reviewer question: would unique
boundary addressing remove the post-marker blind spot?).

Two mechanisms are consistent with the paper's evidence for the blind spot: (a) every boundary
carries the same symbol, so a segment boundary cannot be addressed individually and the position
after it is aliased with the position after every other boundary; or (b) any token standing
between input j-1 and input j interrupts the local lookback, whatever that token is. The three
marker conditions below separate them:

  bar      "|" at every boundary          -- the paper's original marker
  letter   "a" at every boundary          -- a non-digit letter, still the same at every boundary
  ordered  "a", "b", "c", ... in order    -- distinct, ordered, individually addressable labels

Under (a) only `ordered` repairs the post-marker position; under (b) all three leave it broken.
`letter` is the control that separates "it is a bar" from "it is the same symbol every time":
without it, a repair under `ordered` could just mean letters interfere less than bars do.

Tasks: running_xor and running_sum_local (modulus 4), digits, where the blind spot appears; and
random_permutation_digits (alphabet size 8), a position-local control with no blind spot, which
checks that ordered labels keep the within-segment segmentation gain. Lengths N in {32, 64},
k = 4, plus the no-marker baseline. Labels are lowercase letters, which do not collide with the
digit vocabulary of these tasks (unlike a bracketed-digit reminder). Seeds are those of the main
grid (tasks.common.runner._local_seed), so the "bar" cells replicate existing runs.

Scoring follows tasks/common/scoring.py: the marker characters in use and whitespace are removed
from both sides, the prediction is clipped to the truth's length, and positions are compared
index by index. Stored Ground_Truth and LLM_Output are digit-only strings of length N, so the
per-position analysis reads position i mod k == 0 as the first character after a marker.

Output: results/ordered_labels/<model_tag>.csv (one row per generation, appended; resumable)
Usage:
  python scripts/run_ordered_labels.py --model Qwen/Qwen2.5-32B-Instruct [--dry]
"""

import argparse
import csv
import importlib
import random
import re
import string
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.runner import _local_seed, max_new_tokens  # noqa: E402

ORIGINAL = ("Constraint: The input contains vertical bars (|). These are structural boundaries.\n"
            "DO NOT transform them. Copy them exactly to the output.")
LETTERS = string.ascii_lowercase

MARKERS = {
    "bar": ORIGINAL,
    "letter": ("Constraint: The input contains the letter a. These are structural boundaries.\n"
               "DO NOT transform them. Copy them exactly to the output."),
    "ordered": ("Constraint: The input contains lowercase letters (a, b, c, ... in alphabetical\n"
                "order). These are structural boundaries, and each one labels the segment break it\n"
                "marks. DO NOT transform them. Copy them exactly to the output, in the same order."),
}
TASKS = {"running_xor": None, "running_sum_local": 4, "random_permutation_digits": 8}
LENGTHS, KS = [32, 64], [4]


def labels_for(marker, n):
    """The marker string written at each of the n boundaries, in order."""
    if marker == "bar":
        return ["|"] * n
    if marker == "letter":
        return ["a"] * n
    assert marker == "ordered"
    assert n <= len(LETTERS), f"{n} boundaries needs more than {len(LETTERS)} distinct labels"
    return [LETTERS[i] for i in range(n)]


def relabel(s, marker):
    """Replace the i-th '|' of s with the i-th label (positional, not a global replace)."""
    n = s.count("|")
    labs = iter(labels_for(marker, n))
    return re.sub(r"\|", lambda _: next(labs), s)


def marker_chars(marker, n):
    return set(labels_for(marker, n)) | {"|"}


def build(engine, axis, N, k, t, marker):
    rng = random.Random(_local_seed(N, k, axis, t))
    inp, truth, prompt = (engine.build_prompt(N, k, rng, axis_val=axis) if axis
                          else engine.build_prompt(N, k, rng))
    if k == 0:
        return inp, truth, prompt
    head, sep, tail = prompt.rpartition("Input: ")
    assert sep and head.count(ORIGINAL) == 1 and tail.endswith("\nOutput: ")
    assert tail[: -len("\nOutput: ")] == inp
    new_inp = relabel(inp, marker)
    prompt = head.replace(ORIGINAL, MARKERS[marker]) + "Input: " + new_inp + "\nOutput: "
    return new_inp, relabel(truth, marker), prompt


def extract(raw, chars):
    pred = raw.strip().split("Explanation")[0].split("\n")[0]
    for ch in chars | {" ", "\t"}:
        pred = pred.replace(ch, "")
    return pred


def score(raw, truth, chars):
    clean_truth = extract(truth, chars)
    pred = extract(raw, chars)[: len(clean_truth)]
    hits = [i < len(pred) and pred[i] == c for i, c in enumerate(clean_truth)]
    tar = sum(hits) / len(clean_truth) if clean_truth else 0.0
    sar = float(all(hits) and len(pred) == len(clean_truth))
    return pred, clean_truth, tar, sar


def self_check(engines):
    for task, (engine, axis) in engines.items():
        for N in LENGTHS:
            for k in KS:
                nb = (N - 1) // k
                for m in MARKERS:
                    inp, truth, prompt = build(engine, axis, N, k, 0, m)
                    chars = marker_chars(m, nb)
                    _, ct, tar, sar = score(truth, truth, chars)
                    assert tar == 1.0 and sar == 1.0, (task, N, k, m, tar, sar)
                    assert len(ct) == N and ct.isdigit(), (task, N, k, m, ct[:20])
                    got = sum(inp.count(c) for c in set(labels_for(m, nb)))
                    assert got == nb, (task, N, k, m, got, nb)
                    if m == "ordered":            # labels must be distinct and in order
                        seen = [c for c in inp if c in set(LETTERS)]
                        assert seen == labels_for(m, nb), (task, N, k, seen[:8])
    print("Self-check passed: every truth scores 1.0; labels distinct and ordered; truths all digits.")


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
            print(f"----- {m} -----\n{build(e, a, 32, 4, 0, m)[2]}\n")
        print("----- base -----\n" + build(e, a, 32, 0, 0, "bar")[2])
        return

    import torch
    from transformers import AutoTokenizer
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "ordered_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Marker", "Interval", "Seq_Length", "Axis", "Trial", "TAR", "SAR",
              "Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Marker"], r["Interval"], r["Seq_Length"], r["Trial"])
                    for r in csv.DictReader(f)}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()

    plans = [("base", 0)] + [(m, k) for m in MARKERS for k in KS]
    for N in LENGTHS:
        for task, (engine, axis) in engines.items():
            for marker, k in plans:
                nb = (N - 1) // k if k else 0
                chars = marker_chars(marker, nb) if k else {"|"}
                for t in range(args.trials):
                    key = (task, marker, str(k), str(N), str(t))
                    if key in done:
                        continue
                    inp, truth, prompt = build(engine, axis, N, k, t, marker)
                    enc = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                        tokenize=True, add_generation_prompt=True,
                                                        return_tensors="pt", return_dict=True,
                                                        enable_thinking=False)
                    enc = {kk: v.to(model.device) for kk, v in enc.items()}
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=max_new_tokens(engine, N, k),
                                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    pred, ct, tar, sar = score(raw, truth, chars)
                    new = not out_csv.exists()
                    with open(out_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        if new:
                            w.writeheader()
                        w.writerow(dict(Task=task, Marker=marker, Interval=k, Seq_Length=N,
                                        Axis=axis or 2, Trial=t, TAR=tar, SAR=sar, Input_Given=inp,
                                        Ground_Truth=ct, LLM_Output=pred, Raw_Model_Output=raw))
                print(f"done N={N} {task} {marker} k={k}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
