"""Nail down the "sparse boundaries help" finding on genuine (non-degenerate)
attempts only. Discovered that Base ALSO collapses into repetition a lot at
longer lengths (57-70% of trials, not 0% as the tiny original sample
suggested), and that degenerate outputs score misleadingly well by luck.
Excluding degenerate outputs, a real gap survives (Base ~21-37% vs k=8
~35-44% across N=16/32/64), but on small, uneven non-degenerate subsamples
(13-29 per cell). This runs enough EXTRA raw trials, targeted at exactly
the thin cells (Base and k=8 at N=16/32/64 -- k=1 already has 50+
non-degenerate samples everywhere and doesn't need more), to reach ~35-45
genuine non-degenerate samples per cell.

Fresh seeds (trial index 40+), confirmed non-overlapping with every prior
batch (mine 0-19, third-samples 20-39, legacy's separate seed formula).
"""

import os
import random
import sys
from collections import Counter

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from tasks.common import scoring as scoring_mod
from tasks.common.runner import _local_seed, _seed_everything
import tasks.running_xor.engine as running_xor

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# (N, k, num_raw_trials) -- k=0 means Base. Sized from the degenerate-rate
# calculation: enough raw trials to reach ~35-45 non-degenerate samples per cell.
TARGETS = [
    (16, 0, 30), (16, 8, 25),
    (32, 0, 50), (32, 8, 65),
    (64, 0, 70), (64, 8, 115),
]
TRIAL_START = 40


def is_degenerate(s):
    s = str(s)
    if len(s) < 5:
        return False
    counts = Counter(s)
    return counts.most_common(1)[0][1] / len(s) > 0.85


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    _seed_everything()
    grid = []
    for N, k, n_trials in TARGETS:
        for t in range(TRIAL_START, TRIAL_START + n_trials):
            grid.append((N, k, t))
    print(f"Grid size: {len(grid)} evaluations (targeted top-up for non-degenerate sampling)")

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
    topup_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_nondegenerate_topup.csv")
    df.to_csv(topup_csv, index=False)
    print(f"\nSaved {len(df)} rows to {topup_csv}")

    # combine with everything collected so far
    def rescore_stripped(row):
        truth = str(row.Ground_Truth).replace("|", "")
        pred = str(row.LLM_Output).replace("|", "")[: len(truth)]
        n = len(truth)
        matches = sum(1 for i in range(n) if i < len(pred) and pred[i] == truth[i])
        return matches / n if n else 0.0

    mine = pd.read_csv(os.path.join(out_dir, "Qwen2.5-7B-Instruct_FULL_GRID_20trials.csv"))
    third16 = pd.read_csv(os.path.join(out_dir, "Qwen2.5-7B-Instruct_n16_third_sample.csv"))
    third_rest = pd.read_csv(os.path.join(out_dir, "Qwen2.5-7B-Instruct_remaining_third_sample.csv"))
    legacy = pd.read_csv("results/Qwen2.5-7B-Instruct/running_xor_grid_structured_boundaries.csv")
    legacy["TAR"] = legacy.apply(rescore_stripped, axis=1)

    all_data = pd.concat([mine, third16, third_rest, legacy, df], ignore_index=True)
    all_data["degenerate"] = all_data["LLM_Output"].apply(is_degenerate)
    all_data.to_csv(os.path.join(out_dir, "Qwen2.5-7B-Instruct_ALL_DATA_pooled.csv"), index=False)

    print(f"\n{'='*70}\nFINAL: Base vs k=8, GENUINE (non-degenerate) ATTEMPTS ONLY\n{'='*70}")
    clean = all_data[~all_data.degenerate]
    for N in [16, 32, 64]:
        sub = clean[clean.Seq_Length == N]
        base = sub[sub.Method == "Base"]["TAR"]
        base_tar, base_sem = base.mean(), base.std(ddof=1) / (len(base) ** 0.5)
        print(f"\nN={N}: Base TAR={base_tar:.4f} +/- {base_sem:.4f} SEM (n={len(base)} non-degenerate)")
        for k in [1, 8]:
            cb = sub[sub.Interval == k]["TAR"]
            if len(cb) < 2:
                print(f"  k={k}: insufficient non-degenerate samples (n={len(cb)})")
                continue
            cb_sem = cb.std(ddof=1) / (len(cb) ** 0.5)
            delta = cb.mean() - base_tar
            combined_sem = (cb_sem ** 2 + base_sem ** 2) ** 0.5
            ratio = delta / combined_sem if combined_sem > 0 else 0
            sig = "***" if abs(ratio) > 3 else ("**" if abs(ratio) > 2 else ("*" if abs(ratio) > 1 else "no reliable effect"))
            print(f"  k={k}: TAR={cb.mean():.4f}+/-{cb_sem:.4f} (n={len(cb)} non-degenerate)  delta={delta:+.4f}  delta/SEM={ratio:+.2f}  {sig}")


if __name__ == "__main__":
    main()
