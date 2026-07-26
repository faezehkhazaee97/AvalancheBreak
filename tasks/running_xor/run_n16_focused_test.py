"""Focused follow-up: pooled 9-trial data across all lengths showed N=16
as the one length where boundaries beat Base at 4/5 non-trivial intervals
-- but that data came from prompts with added clarifying text, and the
original single-trial verify data showed the OPPOSITE (zero wins) at N=16.
This resolves the contradiction: same N=16, same k grid, ORIGINAL
unmodified prompt (tasks.running_xor.engine, not a variant), 5 fresh
trials/cell for real power, to see whether N=16 is a genuine sweet spot or
an artifact of the prompt-variant text.
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
TRIAL_START = 5
TRIAL_END = 15  # exclusive -- trials 5..14, 10 more, combined with the existing 0..4 gives 15 total


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    _seed_everything()
    grid = [(k, t) for k in INTERVALS for t in range(TRIAL_START, TRIAL_END)]
    print(f"Grid size: {len(grid)} evaluations (N={N} fixed, trials {TRIAL_START}-{TRIAL_END-1}, original unmodified prompt)")

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
        print(f"  k={k} t={t} ({method}): TAR={result['TAR']:.3f}")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", "running_xor", "verify")
    out_csv = os.path.join(out_dir, f"Qwen2.5-7B-Instruct_n16_focused_test_trials{TRIAL_START}-{TRIAL_END-1}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}")

    # combine with the existing trials 0-4 for the full-power analysis
    prior_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_n16_focused_test.csv")
    prior = pd.read_csv(prior_csv)
    combined = pd.concat([prior, df], ignore_index=True)
    combined_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_n16_focused_test_combined.csv")
    combined.to_csv(combined_csv, index=False)
    n_total = combined.Trial.nunique()
    print(f"\nCombined with prior trials 0-4: {len(combined)} total rows ({n_total} trials/cell). Saved to {combined_csv}")

    summary = combined.groupby(["Method", "Interval"])["TAR"].agg(["mean", "std", "count"])
    print(f"\n--- TAR by Method/Interval, N=16, {n_total} trials/cell (COMBINED), ORIGINAL prompt ---")
    print(summary.round(4))

    base = combined[combined.Method == "Base"]["TAR"]
    base_tar, base_sem = base.mean(), base.std(ddof=1) / (len(base) ** 0.5)
    print(f"\nBase TAR (n={len(base)}): {base_tar:.4f} +/- {base_sem:.4f} SEM")
    for k in INTERVALS:
        if k == 0:
            continue
        cb = combined[combined.Interval == k]["TAR"]
        cb_sem = cb.std(ddof=1) / (len(cb) ** 0.5)
        delta = cb.mean() - base_tar
        combined_sem = (cb_sem ** 2 + base_sem ** 2) ** 0.5
        print(f"k={k}: Cellular_Bounds TAR={cb.mean():.4f}+/-{cb_sem:.4f} (n={len(cb)})  "
              f"delta={delta:+.4f}  delta/SEM={delta/combined_sem:+.2f}  "
              f"{'BEATS' if delta > 0 else 'does not beat'} Base")


if __name__ == "__main__":
    main()
