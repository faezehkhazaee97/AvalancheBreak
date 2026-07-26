"""Resolve the N=16 tension: my full-grid 20-trial run found boundaries
robustly HELP at N=16 (k=2,3,4,8 all significant), but the legacy 20-trial
dataset (different random seeds, same prompt) disagreed at k=2 and k=4,
showing boundaries LOSING there. This runs a third, fully independent
20-trial sample (trials 20-39, confirmed non-overlapping with both prior
seed sets) across all of N=16's intervals, so we can see which way a fresh
sample leans and pool all three for a much larger combined N.
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
from tasks.common.runner import _local_seed, _seed_everything
import tasks.running_xor.engine as running_xor

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
N = 16
INTERVALS = [0, 1, 2, 3, 4, 8]
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
    grid = [(k, t) for k in INTERVALS for t in range(TRIAL_START, TRIAL_END)]
    print(f"Grid size: {len(grid)} evaluations (N={N}, trials {TRIAL_START}-{TRIAL_END-1}, third independent sample)")

    rows = []
    for k, t in grid:
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
    third_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_n16_third_sample.csv")
    df.to_csv(third_csv, index=False)
    print(f"\nSaved {len(df)} rows to {third_csv}")

    print(f"\n--- Third independent sample alone, N=16, {TRIAL_END-TRIAL_START} trials/cell ---")
    base3 = df[df.Method == "Base"]["TAR"]
    base3_tar = base3.mean()
    print(f"Base TAR: {base3_tar:.4f} (n={len(base3)})")
    for k in INTERVALS:
        if k == 0:
            continue
        cb = df[df.Interval == k]["TAR"]
        print(f"  k={k}: TAR={cb.mean():.4f} (n={len(cb)})  {'beats' if cb.mean()>base3_tar else 'loses to'} base (this sample alone)")

    # ===== pool all THREE independent samples: mine (t=0-19), legacy (different seed formula), this (t=20-39) =====
    mine = pd.read_csv(os.path.join(out_dir, "Qwen2.5-7B-Instruct_FULL_GRID_20trials.csv"))
    mine_n16 = mine[mine.Seq_Length == 16][["Method", "Interval", "TAR"]]

    legacy_raw = pd.read_csv("results/Qwen2.5-7B-Instruct/running_xor_grid_structured_boundaries.csv")
    legacy_n16 = legacy_raw[legacy_raw.Seq_Length == 16].copy()
    legacy_n16["TAR"] = legacy_n16.apply(_rescore_legacy, axis=1)
    legacy_n16 = legacy_n16[["Method", "Interval", "TAR"]]

    third = df[["Method", "Interval", "TAR"]]

    pooled = pd.concat([mine_n16, legacy_n16, third], ignore_index=True)
    pooled_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_n16_POOLED_3_samples.csv")
    pooled.to_csv(pooled_csv, index=False)

    print(f"\n--- POOLED across all 3 independent samples, N=16 (n up to 60/cell) ---")
    base_p = pooled[pooled.Method == "Base"]["TAR"]
    base_p_tar, base_p_sem = base_p.mean(), base_p.std(ddof=1) / (len(base_p) ** 0.5)
    print(f"Base TAR: {base_p_tar:.4f} +/- {base_p_sem:.4f} SEM (n={len(base_p)})")
    for k in INTERVALS:
        if k == 0:
            continue
        cb = pooled[pooled.Interval == k]["TAR"]
        cb_sem = cb.std(ddof=1) / (len(cb) ** 0.5)
        delta = cb.mean() - base_p_tar
        combined_sem = (cb_sem ** 2 + base_p_sem ** 2) ** 0.5
        sig = "***" if abs(delta / combined_sem) > 3 else ("**" if abs(delta / combined_sem) > 2 else ("*" if abs(delta / combined_sem) > 1 else ""))
        print(f"  k={k}: TAR={cb.mean():.4f}+/-{cb_sem:.4f} (n={len(cb)})  delta={delta:+.4f}  delta/SEM={delta/combined_sem:+.2f} {sig}")


def _rescore_legacy(row):
    truth = str(row.Ground_Truth).replace("|", "")
    pred = str(row.LLM_Output).replace("|", "")[: len(truth)]
    n = len(truth)
    matches = sum(1 for i in range(n) if i < len(pred) and pred[i] == truth[i])
    return matches / n if n else 0.0


if __name__ == "__main__":
    main()
