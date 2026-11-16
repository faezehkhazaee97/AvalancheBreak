#!/usr/bin/env python3
"""Two reviewer questions about what the delimiter has to be, on one grid.

(1) Does the segmentation gain need a delimiter the model must COPY, or is an out-of-band
    separator enough? The one-sided ablation says marking the input alone hurts unless the
    output is marked too, which suggests copying matters. A single space is the natural test: it
    separates the input without adding a token the answer has to reproduce, and the scorer strips
    whitespace from both sides anyway.

      space_both  a space every k characters in the input, and the answer written the same way
      space_in    a space every k characters in the input, the answer written with no separator

(2) Does the spacing have to be fixed? The paper sweeps k at fixed N; here k grows with the
    sequence so the number of segments is held at eight instead, which is the "adaptive k" a
    reviewer asked about.

      bar_adaptive  "|" every N/8 characters, both sides

  bar_k4 is the paper's own condition at this k, carried along as the reference, and base is the
  no-delimiter run. Tasks, seeds, lengths and decoding follow scripts/run_ordered_labels.py, so
  the bar_k4 cells replicate existing numbers.

Scoring strips the delimiter in use and whitespace from both sides, as elsewhere, so a run whose
answer carries no separator is not penalised for that.

Output: results/delimiter_variants/<model_tag>.csv (one row per generation, appended; resumable)
Usage:  python scripts/run_delimiter_variants.py --model Qwen/Qwen2.5-32B-Instruct [--dry]
"""

import argparse
import csv
import importlib
import math
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.runner import _local_seed, max_new_tokens  # noqa: E402

ORIGINAL = ("Constraint: The input contains vertical bars (|). These are structural boundaries.\n"
            "DO NOT transform them. Copy them exactly to the output.")
SPACE_BOTH = ("Constraint: The input is written in groups separated by single spaces. These are\n"
              "structural boundaries. Keep the same grouping in your output.")
SPACE_IN = ("Constraint: The input is written in groups separated by single spaces. These are\n"
            "structural boundaries. Do NOT put spaces in your output: write the answer as one\n"
            "unbroken run of characters.")

TASKS = {"running_xor": None, "running_sum_local": 4, "random_permutation_digits": 8}
LENGTHS = [32, 64]
SEGMENTS = 8                      # bar_adaptive holds the segment count, not the spacing


def spacing(cond, N):
    if cond == "base":
        return 0
    if cond == "bar_adaptive":
        return max(2, N // SEGMENTS)
    return 4


def render(inp_bars, ans_bars, cond):
    """Rewrite the engine's bar-delimited strings into the condition's own format."""
    if cond == "base":
        return inp_bars, ans_bars
    if cond in ("bar_k4", "bar_adaptive"):
        return inp_bars, ans_bars
    sp_in = inp_bars.replace("|", " ")
    sp_ans = ans_bars.replace("|", " ") if cond == "space_both" else ans_bars.replace("|", "")
    return sp_in, sp_ans


CONSTRAINT = {"bar_k4": ORIGINAL, "bar_adaptive": ORIGINAL,
              "space_both": SPACE_BOTH, "space_in": SPACE_IN}
# bar_adaptive (k = N/8) was dropped before running: at N = 32 it is k = 4, the reference,
# and at N = 64 it is k = 8, which the main k-sweep already measures, so it would have been
# redundant GPU time. The adaptive-spacing question is answered from that existing grid in
# scripts/analyze_adaptive_k.py instead.
CONDS = ["base", "bar_k4", "space_both", "space_in"]


def build(engine, axis, cond, N, t):
    k = spacing(cond, N)
    rng = random.Random(_local_seed(N, k, axis, t))
    inp, truth, prompt = (engine.build_prompt(N, k, rng, axis_val=axis) if engine.AXIS_NAME
                          else engine.build_prompt(N, k, rng))
    if cond == "base":
        return inp, truth, prompt, k
    head, sep, tail = prompt.rpartition("Input: ")
    assert sep and head.count(ORIGINAL) == 1 and tail.endswith("\nOutput: ")
    assert tail[: -len("\nOutput: ")] == inp
    new_inp, new_ans = render(inp, truth, cond)
    prompt = head.replace(ORIGINAL, CONSTRAINT[cond]) + "Input: " + new_inp + "\nOutput: "
    return new_inp, new_ans, prompt, k


def clean(s):
    return re.sub(r"[|\s]", "", str(s))


def score(raw, truth):
    ct = clean(truth)
    pred = clean(raw.strip().split("Explanation")[0].split("\n")[0])[: len(ct)]
    hits = [i < len(pred) and pred[i] == c for i, c in enumerate(ct)]
    tar = sum(hits) / len(ct) if ct else 0.0
    return pred, ct, tar, float(all(hits) and len(pred) == len(ct))


def self_check(engines):
    for task, (e, a) in engines.items():
        for N in LENGTHS:
            for cond in CONDS:
                inp, ans, prompt, k = build(e, a, cond, N, 0)
                _, ct, tar, sar = score(ans, ans)
                assert tar == 1.0 and sar == 1.0 and len(ct) == N, (task, N, cond, tar, len(ct))
                if cond == "space_in":
                    assert " " not in ans and "|" not in ans, (task, N, "answer must be unbroken")
                if cond == "space_both":
                    assert ans.count(" ") == (N - 1) // k, (task, N, cond)
                if cond == "bar_adaptive":
                    assert k == max(2, N // SEGMENTS), (task, N, k)
    print("Self-check passed: every truth scores 1.0; separators where the condition says.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    engines = {t: (importlib.import_module(f"tasks.{t}.engine"), a) for t, a in TASKS.items()}
    self_check(engines)
    if args.dry:
        e, a = engines["running_xor"]
        for cond in CONDS:
            print(f"----- {cond} -----\n{build(e, a, cond, 32, 0)[2]}\n")
        print(len(TASKS) * len(LENGTHS) * len(CONDS) * args.trials, "generations")
        return

    import torch
    from transformers import AutoTokenizer
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "delimiter_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Condition", "Interval", "Seq_Length", "Axis", "Trial", "TAR", "SAR",
              "Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Condition"], r["Seq_Length"], r["Trial"]) for r in csv.DictReader(f)}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()

    for N in LENGTHS:
        for task, (engine, axis) in engines.items():
            for cond in CONDS:
                for t in range(args.trials):
                    key = (task, cond, str(N), str(t))
                    if key in done:
                        continue
                    inp, ans, prompt, k = build(engine, axis, cond, N, t)
                    enc = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                        tokenize=True, add_generation_prompt=True,
                                                        return_tensors="pt", return_dict=True,
                                                        enable_thinking=False)
                    enc = {kk: v.to(model.device) for kk, v in enc.items()}
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=max_new_tokens(engine, N, max(k, 1)),
                                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    pred, ct, tar, sar = score(raw, ans)
                    new = not out_csv.exists()
                    with open(out_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        if new:
                            w.writeheader()
                        w.writerow(dict(Task=task, Condition=cond, Interval=k, Seq_Length=N,
                                        Axis=axis or 2, Trial=t, TAR=tar, SAR=sar,
                                        Input_Given=inp, Ground_Truth=ct, LLM_Output=pred,
                                        Raw_Model_Output=raw))
                print(f"done N={N} {task} {cond}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
