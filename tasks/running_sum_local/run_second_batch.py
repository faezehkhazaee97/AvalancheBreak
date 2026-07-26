"""Second, INDEPENDENT batch (trials 40-79, confirmed non-overlapping with
the first batch's trials 0-39) for running_sum_local on a given model --
same verification standard already applied to running_xor: many of the
first-batch deltas were only 1-2 SEM (the kind that evaporated on XOR's
second batch), so nothing gets trusted until it survives pooling with an
independent sample.
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
import tasks.running_sum_local.engine as running_sum_local

TRIAL_START = 40
TRIAL_END = 80
MODULI = running_sum_local.AXIS_VALUES


def is_degenerate(s):
    s = str(s)
    if len(s) < 5:
        return False
    counts = Counter(s)
    return counts.most_common(1)[0][1] / len(s) > 0.85


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
    model_tag = model_id.split("/")[-1]

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    _seed_everything()
    grid = [(N, k, M, t) for N in STANDARD_LENGTHS for k in valid_intervals(N, STANDARD_INTERVALS)
            for M in MODULI for t in range(TRIAL_START, TRIAL_END)]
    print(f"Grid size: {len(grid)} evaluations (SECOND independent batch, trials {TRIAL_START}-{TRIAL_END-1}, model={model_tag})")

    rows = []
    for idx, (N, k, M, t) in enumerate(grid):
        method = "Base" if k == 0 else "Cellular_Bounds"
        seed = _local_seed(N, k, M, t)
        rng = random.Random(seed)
        input_str, truth, prompt = running_sum_local.build_prompt(N, k, rng, axis_val=M)

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
            "Method": method, "Interval": k, "Seq_Length": N, "Modulus": M, "Trial": t,
            "Input_Given": input_str, "Ground_Truth": result["clean_truth"],
            "LLM_Output": result["clean_pred"], "TAR": result["TAR"], "SAR": result["SAR"],
            "Degenerate": is_degenerate(result["clean_pred"]), "Raw_Model_Output": raw_output,
        })
        if idx % 300 == 0:
            print(f"  [{idx}/{len(grid)}] N={N} k={k} M={M} t={t}: TAR={result['TAR']:.3f}")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", "running_sum_local", "verify")
    os.makedirs(out_dir, exist_ok=True)
    second_csv = os.path.join(out_dir, f"{model_tag}_second_batch_40trials.csv")
    df.to_csv(second_csv, index=False)
    print(f"\nSaved {len(df)} rows to {second_csv}")

    # pool both batches (all columns, not just TAR -- needed for SAR/degenerate plotting too)
    first = pd.read_csv(os.path.join(out_dir, f"{model_tag}_FULL_GRID_40trials.csv"))
    overlap = set(zip(first.Seq_Length, first.Interval, first.Modulus, first.Trial)) & \
              set(zip(df.Seq_Length, df.Interval, df.Modulus, df.Trial))
    assert len(overlap) == 0, f"non-independent trials detected: {overlap}"

    pooled = pd.concat([first, df], ignore_index=True)
    pooled_csv = os.path.join(out_dir, f"{model_tag}_POOLED_80trials.csv")
    pooled.to_csv(pooled_csv, index=False)
    print(f"Saved pooled n={len(pooled)} to {pooled_csv}")

    print(f"\n{'='*78}\nPOOLED (both batches, n=80/cell): {model_tag}\n{'='*78}")
    for M in MODULI:
        print(f"\n===== Modulus={M} =====")
        for N in STANDARD_LENGTHS:
            sub = pooled[(pooled.Seq_Length == N) & (pooled.Modulus == M)]
            base = sub[sub.Method == "Base"]["TAR"]
            base_tar, base_sem = base.mean(), base.std(ddof=1) / (len(base) ** 0.5)
            print(f"N={N}: Base TAR={base_tar:.4f}+/-{base_sem:.4f} (n={len(base)}), degenerate={sub[sub.Method=='Base'].Degenerate.mean():.1%}")
            for k in sorted(sub[sub.Method == "Cellular_Bounds"].Interval.unique()):
                cb = sub[sub.Interval == k]["TAR"]
                cb_sem = cb.std(ddof=1) / (len(cb) ** 0.5)
                delta = cb.mean() - base_tar
                combined_sem = (cb_sem ** 2 + base_sem ** 2) ** 0.5
                ratio = delta / combined_sem if combined_sem > 0 else 0
                sig = "***" if abs(ratio) > 3 else ("**" if abs(ratio) > 2 else ("*" if abs(ratio) > 1 else "ns"))
                print(f"  k={k}: TAR={cb.mean():.4f} (n={len(cb)})  delta={delta:+.4f}  delta/SEM={ratio:+.2f} {sig}  (degenerate={sub[sub.Interval==k].Degenerate.mean():.1%})")


if __name__ == "__main__":
    main()
