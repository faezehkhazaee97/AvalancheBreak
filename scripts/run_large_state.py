#!/usr/bin/env python3
"""Does the blind spot survive a larger state space? (Reviewer question: all dependency tasks
use alphabets or moduli of at most eight symbols.)

\\task{running_sum_local} at modulus M in {8, 16, 32} on Qwen2.5-32B: output[i] is the symbol
whose value is (value[i] + value[i-1]) mod M, with the first symbol copied. Symbols are the
single characters "0123456789abc..."[:M], written space-separated so that each is its own token
at every M (the spaced convention of tasks/common/spaced.py); scoring strips spaces and markers
and compares character by character, as everywhere else.

M = 8 is run here too, in the same spaced format, as the within-experiment baseline: the main
grid's M = 8 cells use unspaced digits, so they are not directly comparable.

N in {32, 64}, no markers and "|" every k in {4, 16}, 20 trials per cell, greedy decoding.
Budget 2N + 40 new tokens, more generous than the main grid's 1.5N + 20 because the spaced
format needs about one token per symbol; predictions are clipped to the answer's length before
scoring, so a larger budget cannot raise accuracy, it only avoids truncation.

Output: results/large_state/<model_tag>.csv
Usage:  python scripts/run_large_state.py --model Qwen/Qwen2.5-32B-Instruct [--dry]
"""

import argparse
import csv
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common import scoring  # noqa: E402
from tasks.common.boundary import insert_boundaries  # noqa: E402
from tasks.common.runner import _local_seed  # noqa: E402

SYMBOLS = "0123456789abcdefghijklmnopqrstuv"  # 32 single characters
MODULI = [8, 16, 32]
LENGTHS, KS = [32, 64], [0, 4, 16]


def build(N, k, M, t):
    rng = random.Random(_local_seed(N, k, M, t))
    sym = SYMBOLS[:M]
    vals = [rng.randrange(M) for _ in range(N)]
    out_vals = [vals[0]] + [(vals[i] + vals[i - 1]) % M for i in range(1, N)]
    inp = insert_boundaries("".join(sym[v] for v in vals), k)
    out = insert_boundaries("".join(sym[v] for v in out_vals), k)
    listing = " ".join(sym)
    ex_in = [rng.randrange(M) for _ in range(4)]
    ex_out = [ex_in[0]] + [(ex_in[i] + ex_in[i - 1]) % M for i in range(1, 4)]
    prompt = f"""You are a deterministic sequence transformer.
Task: Running sum with the previous symbol (local window).
Alphabet: {listing} ({M} symbols, with values 0 to {M - 1} in this order)
Modulus: {M}
Rule: At each position, output the symbol whose value is (value of the current symbol + value of the previous symbol) mod {M}.
The first symbol is copied as-is (no previous symbol to add).
Each subsequent symbol: value(output[i]) = (value(input[i]) + value(input[i-1])) mod {M}
Example: Input {' '.join(sym[v] for v in ex_in)} -> Output {' '.join(sym[v] for v in ex_out)}
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
Write the output in the same format as the input: every character separated by a single space. No explanations.

Input: {' '.join(inp)}
Output: """
    if k == 0:  # no markers: drop the marker constraint, as the main grid's Base does not
        prompt = prompt.replace("Constraint: The input contains vertical bars (|). These are structural boundaries.\n"
                                "DO NOT transform them. Copy them exactly to the output.\n", "")
    return " ".join(inp), out, prompt


def self_check():
    for M in MODULI:
        for N in LENGTHS:
            for k in KS:
                inp, truth, prompt = build(N, k, M, 0)
                r = scoring.score(" ".join(truth), truth)
                assert r["TAR"] == 1.0 and len(r["clean_truth"]) == N, (M, N, k)
                assert inp.replace(" ", "").count("|") == (0 if k == 0 else (N - 1) // k), (M, N, k)
                assert prompt.count("Constraint:") == (0 if k == 0 else 1)
    print("Self-check passed: every truth scores 1.0; marker counts and prompts are as expected.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    self_check()
    if args.dry:
        print(build(32, 4, 16, 0)[2])
        print(len(MODULI) * len(LENGTHS) * len(KS) * args.trials, "generations")
        return

    import torch
    from transformers import AutoTokenizer
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "large_state"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Modulus", "Method", "Interval", "Seq_Length", "Trial", "TAR", "SAR",
              "Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Modulus"], r["Interval"], r["Seq_Length"], r["Trial"]) for r in csv.DictReader(f)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()
    for M in MODULI:
        for N in LENGTHS:
            for k in KS:
                for t in range(args.trials):
                    if (str(M), str(k), str(N), str(t)) in done:
                        continue
                    inp, truth, prompt = build(N, k, M, t)
                    enc = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                       add_generation_prompt=True, return_tensors="pt",
                                                       return_dict=True, enable_thinking=False)
                    enc = {kk: v.to(model.device) for kk, v in enc.items()}
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=2 * N + 40, do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)
                    raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    res = scoring.score(raw, truth)
                    new = not out_csv.exists()
                    with open(out_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        if new:
                            w.writeheader()
                        w.writerow(dict(Task="running_sum_local_spaced", Modulus=M,
                                        Method="Base" if k == 0 else "Cellular_Bounds", Interval=k, Seq_Length=N,
                                        Trial=t, TAR=res["TAR"], SAR=res["SAR"], Input_Given=inp,
                                        Ground_Truth=res["clean_truth"], LLM_Output=res["clean_pred"],
                                        Raw_Model_Output=raw))
                print(f"done M={M} N={N} k={k}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
