#!/usr/bin/env python3
"""Do the long-sequence gain and the post-marker blind spot survive sampling? (Reviewer question;
every other run in the paper decodes greedily.)

Qwen2.5-32B on random_permutation_digits (A = 8; position-local), running_xor and
running_sum_local (modulus 4; dependency), N in {32, 64}, no markers and "|" every k in {4, 16}.
Same prompts and seeds as the main grid (tasks.common.runner._local_seed), so the greedy
reference is the existing main-grid data. Two sampling regimes, one sample per prompt:
  t07: temperature 0.7, top-p 0.8   (the temperature and top-p of the model's shipped
                                     generation config; no top-k, no repetition penalty)
  t10: temperature 1.0, top-p 0.95
The sampling seed is set per generation, so runs are reproducible and resumable.
Scoring is tasks/common/scoring.py, unchanged.

Output: results/sampling_decoding/<model_tag>.csv
Usage:  python scripts/run_sampling_decoding.py --model Qwen/Qwen2.5-32B-Instruct [--dry]
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
from tasks.common.runner import _local_seed, max_new_tokens  # noqa: E402

TASKS = {"random_permutation_digits": 8, "running_xor": None, "running_sum_local": 4}
LENGTHS, KS = [32, 64], [0, 4, 16]
REGIMES = {"t07": dict(temperature=0.7, top_p=0.8), "t10": dict(temperature=1.0, top_p=0.95)}


def build(engine, axis, N, k, t):
    rng = random.Random(_local_seed(N, k, axis, t))
    return engine.build_prompt(N, k, rng, axis_val=axis) if axis else engine.build_prompt(N, k, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    engines = {t: (importlib.import_module(f"tasks.{t}.engine"), a) for t, a in TASKS.items()}
    if args.dry:
        for t, (e, a) in engines.items():
            inp, truth, prompt = build(e, a, 32, 4, 0)
            assert scoring.score(truth, truth)["TAR"] == 1.0
            print(t, inp, "->", truth)
        print(len(TASKS) * len(LENGTHS) * len(KS) * args.trials * len(REGIMES), "generations")
        return

    import torch
    from transformers import AutoTokenizer
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "sampling_decoding"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Regime", "Method", "Interval", "Seq_Length", "Axis", "Trial", "TAR", "SAR",
              "Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Regime"], r["Interval"], r["Seq_Length"], r["Trial"]) for r in csv.DictReader(f)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()
    for ri, (regime, kw) in enumerate(REGIMES.items()):
        for task, (engine, axis) in engines.items():
            for N in LENGTHS:
                for k in KS:
                    for t in range(args.trials):
                        key = (task, regime, str(k), str(N), str(t))
                        if key in done:
                            continue
                        inp, truth, prompt = build(engine, axis, N, k, t)
                        enc = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                           add_generation_prompt=True, return_tensors="pt",
                                                           return_dict=True, enable_thinking=False)
                        enc = {kk: v.to(model.device) for kk, v in enc.items()}
                        torch.manual_seed(1_000_003 * (ri + 1) + _local_seed(N, k, axis, t))
                        with torch.no_grad():
                            out = model.generate(**enc, max_new_tokens=max_new_tokens(engine, N, k), do_sample=True,
                                                 top_k=0, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id,
                                                 **kw)
                        raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                        res = scoring.score(raw, truth)
                        new = not out_csv.exists()
                        with open(out_csv, "a", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=fields)
                            if new:
                                w.writeheader()
                            w.writerow(dict(Task=task, Regime=regime, Method="Base" if k == 0 else "Cellular_Bounds",
                                            Interval=k, Seq_Length=N, Axis=axis or 2, Trial=t, TAR=res["TAR"],
                                            SAR=res["SAR"], Input_Given=inp, Ground_Truth=res["clean_truth"],
                                            LLM_Output=res["clean_pred"], Raw_Model_Output=raw))
                    print(f"done {regime} {task} N={N} k={k}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
