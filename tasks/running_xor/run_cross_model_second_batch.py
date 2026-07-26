"""Second, INDEPENDENT batch (trials 40-79, confirmed non-overlapping with
the first batch's trials 0-39) for a given model, full running_xor grid --
same verification standard already applied to the 7B data, now extended to
14B/32B before trusting their first-batch results.
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
from tasks.common.boundary import valid_intervals, STANDARD_INTERVALS, STANDARD_LENGTHS
from tasks.common.runner import _local_seed, _seed_everything
import tasks.running_xor.engine as running_xor

TRIAL_START = 40
TRIAL_END = 80


def is_degenerate(s):
    s = str(s)
    if len(s) < 5:
        return False
    counts = Counter(s)
    return counts.most_common(1)[0][1] / len(s) > 0.85


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-14B-Instruct"
    model_tag = model_id.split("/")[-1]

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    _seed_everything()
    grid = [(N, k, t) for N in STANDARD_LENGTHS for k in valid_intervals(N, STANDARD_INTERVALS)
            for t in range(TRIAL_START, TRIAL_END)]
    print(f"Grid size: {len(grid)} evaluations (SECOND independent batch, trials {TRIAL_START}-{TRIAL_END-1}, model={model_tag})")

    rows = []
    for idx, (N, k, t) in enumerate(grid):
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
            "Degenerate": is_degenerate(result["clean_pred"]), "Raw_Model_Output": raw_output,
        })
        if idx % 300 == 0:
            print(f"  [{idx}/{len(grid)}] N={N} k={k} t={t}: TAR={result['TAR']:.3f}")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", "running_xor", "verify")
    second_csv = os.path.join(out_dir, f"{model_tag}_second_batch_40trials.csv")
    df.to_csv(second_csv, index=False)
    print(f"\nSaved {len(df)} rows to {second_csv}")

    # pool both batches for the real, verified answer
    first = pd.read_csv(os.path.join(out_dir, f"{model_tag}_FULL_GRID_40trials.csv"))
    cols = ["Method", "Interval", "Seq_Length", "TAR"]
    pooled = pd.concat([first[cols], df[cols]], ignore_index=True)
    pooled.to_csv(os.path.join(out_dir, f"{model_tag}_POOLED_80trials.csv"), index=False)

    print(f"\n{'='*70}\nPOOLED (both batches, n=80/cell): {model_tag}\n{'='*70}")
    for N in STANDARD_LENGTHS:
        sub = pooled[pooled.Seq_Length == N]
        base = sub[sub.Method == "Base"]["TAR"]
        base_tar, base_sem = base.mean(), base.std(ddof=1) / (len(base) ** 0.5)
        print(f"\nN={N}: Base TAR={base_tar:.4f} +/- {base_sem:.4f} (n={len(base)})")
        for k in sorted(sub[sub.Method == "Cellular_Bounds"].Interval.unique()):
            cb = sub[sub.Interval == k]["TAR"]
            cb_sem = cb.std(ddof=1) / (len(cb) ** 0.5)
            delta = cb.mean() - base_tar
            combined_sem = (cb_sem ** 2 + base_sem ** 2) ** 0.5
            ratio = delta / combined_sem if combined_sem > 0 else 0
            sig = "***" if abs(ratio) > 3 else ("**" if abs(ratio) > 2 else ("*" if abs(ratio) > 1 else "no reliable effect"))
            print(f"  k={k}: TAR={cb.mean():.4f} (n={len(cb)})  delta={delta:+.4f}  delta/SEM={ratio:+.2f}  {sig}")


if __name__ == "__main__":
    main()
