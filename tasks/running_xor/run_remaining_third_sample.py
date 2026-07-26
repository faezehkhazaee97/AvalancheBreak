"""Bring N=4, 8, 32, 64 up to the same 3-independent-seed confidence level
already established for N=16 (which needed a third sample to catch that two
of its five conditions were actually noise). Third batch of trials
(20-39), confirmed non-overlapping with both the original run (0-19) and
the legacy dataset's different seed formula.
"""

import os
import random
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from tasks.common import scoring as scoring_mod
from tasks.common.boundary import valid_intervals, STANDARD_INTERVALS
from tasks.common.runner import _local_seed, _seed_everything
import tasks.running_xor.engine as running_xor

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LENGTHS = [4, 8, 32, 64]
TRIAL_START = 20
TRIAL_END = 40


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    _seed_everything()
    grid = [(N, k, t) for N in LENGTHS for k in valid_intervals(N, STANDARD_INTERVALS)
            for t in range(TRIAL_START, TRIAL_END)]
    print(f"Grid size: {len(grid)} evaluations (N in {LENGTHS}, trials {TRIAL_START}-{TRIAL_END-1}, third independent sample)")

    rows = []
    for N, k, t in grid:
        method = "Base" if k == 0 else "Cellular_Bounds"
        seed = _local_seed(N, k, None, t)
        rng = random.Random(seed)
        input_str, truth, prompt = running_xor.build_prompt(N, k, rng)

        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}
        max_new = int(N * 1.5) + 20

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_only = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)
        result = scoring_mod.score(raw_output, truth)

        rows.append({
            "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
            "Input_Given": input_str, "Ground_Truth": result["clean_truth"],
            "LLM_Output": result["clean_pred"], "TAR": result["TAR"], "SAR": result["SAR"],
            "Raw_Model_Output": raw_output,
        })

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", "running_xor", "verify")
    third_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_remaining_third_sample.csv")
    df.to_csv(third_csv, index=False)
    print(f"\nSaved {len(df)} rows to {third_csv}")

    # pool: mine (0-19) + legacy (different formula) + this new third sample (20-39)
    def rescore_stripped(row):
        truth = str(row.Ground_Truth).replace("|", "")
        pred = str(row.LLM_Output).replace("|", "")[: len(truth)]
        n = len(truth)
        matches = sum(1 for i in range(n) if i < len(pred) and pred[i] == truth[i])
        return matches / n if n else 0.0

    mine = pd.read_csv(os.path.join(out_dir, "Qwen2.5-7B-Instruct_FULL_GRID_20trials.csv"))
    legacy = pd.read_csv("results/Qwen2.5-7B-Instruct/running_xor_grid_structured_boundaries.csv")
    legacy["TAR"] = legacy.apply(rescore_stripped, axis=1)

    print(f"\n{'='*70}\nFINAL: pooled across 3 independent seed sets, N in {LENGTHS}\n{'='*70}")
    for N in LENGTHS:
        valid_k = valid_intervals(N, STANDARD_INTERVALS)
        m = mine[(mine.Seq_Length == N) & (mine.Interval.isin(valid_k))][["Method", "Interval", "TAR"]]
        l = legacy[(legacy.Seq_Length == N) & (legacy.Interval.isin(valid_k))][["Method", "Interval", "TAR"]]
        t3 = df[df.Seq_Length == N][["Method", "Interval", "TAR"]]
        pooled = pd.concat([m, l, t3], ignore_index=True)

        base = pooled[pooled.Method == "Base"]["TAR"]
        base_tar, base_sem = base.mean(), base.std(ddof=1) / (len(base) ** 0.5)
        print(f"\nN={N}: pooled Base TAR={base_tar:.4f} +/- {base_sem:.4f} (n={len(base)})")
        for k in sorted(valid_k):
            if k == 0:
                continue
            cb = pooled[pooled.Interval == k]["TAR"]
            cb_sem = cb.std(ddof=1) / (len(cb) ** 0.5)
            delta = cb.mean() - base_tar
            combined_sem = (cb_sem ** 2 + base_sem ** 2) ** 0.5
            ratio = delta / combined_sem if combined_sem > 0 else 0
            sig = "***" if abs(ratio) > 3 else ("**" if abs(ratio) > 2 else ("*" if abs(ratio) > 1 else "no reliable effect"))
            print(f"  k={k}: TAR={cb.mean():.4f} (n={len(cb)})  delta={delta:+.4f}  delta/SEM={ratio:+.2f}  {sig}")


if __name__ == "__main__":
    main()
