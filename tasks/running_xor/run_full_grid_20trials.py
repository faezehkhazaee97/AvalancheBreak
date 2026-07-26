"""Complete the running_xor production grid for Qwen2.5-7B-Instruct at a
full 20 trials/cell -- this is exactly the running_xor portion of the
production run from the original clean-pipeline plan, just built up
incrementally: we already have 15 trials/cell (from the N=16 test + the
N=4/8/32/64 test), so this runs the remaining 5 (trials 15-19) across the
FULL grid (all 5 lengths this time, including N=16) and combines everything
into one final, complete dataset.

Original unmodified prompt (tasks.running_xor.engine), same seed formula
throughout, so all 20 trials/cell are genuinely paired with what a full
production run would have produced.
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
from tasks.common.boundary import valid_intervals, STANDARD_INTERVALS, STANDARD_LENGTHS
from tasks.common.runner import _local_seed, _seed_everything
import tasks.running_xor.engine as running_xor

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TRIAL_START = 15
TRIAL_END = 20


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    _seed_everything()
    grid = [(N, k, t) for N in STANDARD_LENGTHS for k in valid_intervals(N, STANDARD_INTERVALS)
            for t in range(TRIAL_START, TRIAL_END)]
    print(f"Grid size: {len(grid)} evaluations (full grid, trials {TRIAL_START}-{TRIAL_END-1})")

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
    topup_csv = os.path.join(out_dir, f"Qwen2.5-7B-Instruct_topup_trials{TRIAL_START}-{TRIAL_END-1}.csv")
    df.to_csv(topup_csv, index=False)
    print(f"\nSaved {len(df)} top-up rows to {topup_csv}")

    # combine with everything already collected: N=16 (t=0-14) + N=4/8/32/64 (t=0-14) + this top-up (t=15-19)
    n16 = pd.read_csv(os.path.join(out_dir, "Qwen2.5-7B-Instruct_n16_focused_test_combined.csv"))
    multi = pd.read_csv(os.path.join(out_dir, "Qwen2.5-7B-Instruct_multi_length_focused_test.csv"))
    combined = pd.concat([n16, multi, df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Seq_Length", "Interval", "Trial"], keep="first")

    final_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_FULL_GRID_20trials.csv")
    combined.to_csv(final_csv, index=False)
    print(f"\nFinal combined dataset: {len(combined)} rows. Saved to {final_csv}")

    counts = combined.groupby(["Seq_Length", "Interval"]).size().unstack()
    print("\n--- trial count per cell (should all be 20) ---")
    print(counts)

    print(f"\n--- FINAL: TAR by Length/Interval, full grid, 20 trials/cell, ORIGINAL prompt ---")
    for N in STANDARD_LENGTHS:
        sub = combined[combined.Seq_Length == N]
        base = sub[sub.Method == "Base"]["TAR"]
        base_tar, base_sem = base.mean(), base.std(ddof=1) / (len(base) ** 0.5)
        print(f"\nN={N}: Base TAR={base_tar:.4f} +/- {base_sem:.4f} SEM (n={len(base)})")
        for k in sorted(sub[sub.Method == "Cellular_Bounds"].Interval.unique()):
            cb = sub[sub.Interval == k]["TAR"]
            cb_sem = cb.std(ddof=1) / (len(cb) ** 0.5)
            delta = cb.mean() - base_tar
            combined_sem = (cb_sem ** 2 + base_sem ** 2) ** 0.5
            sig = "***" if abs(delta / combined_sem) > 3 else ("**" if abs(delta / combined_sem) > 2 else ("*" if abs(delta / combined_sem) > 1 else ""))
            print(f"  k={k}: TAR={cb.mean():.4f}+/-{cb_sem:.4f} (n={len(cb)})  delta={delta:+.4f}  delta/SEM={delta/combined_sem:+.2f} {sig}")


if __name__ == "__main__":
    main()
