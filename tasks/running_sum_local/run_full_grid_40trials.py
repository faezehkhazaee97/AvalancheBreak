"""Full production-style run of running_sum_local: all 5 lengths, all valid
intervals, all 3 moduli (2, 4, 8 -- modulus 2 is the direct XOR-equivalent
case), 40 trials/cell, on a given model. Same verified engine/scoring/grid
code audited and confirmed correct before this run (hand-checked math,
confirmed the Finding-A truncation bug cannot occur since scoring is fully
delegated to tasks.common.scoring, confirmed no degenerate k>=N cells).
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
from tasks.common.runner import _build_grid, _local_seed, _seed_everything
import tasks.running_sum_local.engine as running_sum_local

TRIALS_PER_CELL = 40


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
    grid = _build_grid(running_sum_local, trials_per_cell=TRIALS_PER_CELL)
    print(f"Grid size: {len(grid)} evaluations (full grid x 3 moduli, {TRIALS_PER_CELL} trials/cell, model={model_tag})")

    rows = []
    for idx, (N, k, axis_val, t) in enumerate(grid):
        method = "Base" if k == 0 else "Cellular_Bounds"
        seed = _local_seed(N, k, axis_val, t)
        rng = random.Random(seed)
        input_str, truth, prompt = running_sum_local.build_prompt(N, k, rng, axis_val=axis_val)

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
            "Method": method, "Interval": k, "Seq_Length": N, "Modulus": axis_val, "Trial": t,
            "Input_Given": input_str, "Ground_Truth": result["clean_truth"],
            "LLM_Output": result["clean_pred"], "TAR": result["TAR"], "SAR": result["SAR"],
            "Degenerate": is_degenerate(result["clean_pred"]), "Raw_Model_Output": raw_output,
        })
        if idx % 300 == 0:
            print(f"  [{idx}/{len(grid)}] N={N} k={k} M={axis_val} t={t}: TAR={result['TAR']:.3f}")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", "running_sum_local", "verify")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{model_tag}_FULL_GRID_40trials.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}")

    print(f"\n--- TAR by Length/Modulus/Interval, {model_tag}, {TRIALS_PER_CELL} trials/cell ---")
    for M in [2, 4, 8]:
        print(f"\n===== Modulus={M} =====")
        for N in sorted(df.Seq_Length.unique()):
            sub = df[(df.Seq_Length == N) & (df.Modulus == M)]
            base = sub[sub.Method == "Base"]["TAR"]
            if len(base) == 0:
                continue
            base_tar, base_sem = base.mean(), base.std(ddof=1) / (len(base) ** 0.5)
            print(f"N={N}: Base TAR={base_tar:.4f}+/-{base_sem:.4f} (n={len(base)}), degenerate={sub[sub.Method=='Base'].Degenerate.mean():.1%}")
            for k in sorted(sub[sub.Method == "Cellular_Bounds"].Interval.unique()):
                cb = sub[sub.Interval == k]["TAR"]
                cb_sem = cb.std(ddof=1) / (len(cb) ** 0.5)
                delta = cb.mean() - base_tar
                combined_sem = (cb_sem ** 2 + base_sem ** 2) ** 0.5
                ratio = delta / combined_sem if combined_sem > 0 else 0
                sig = "***" if abs(ratio) > 3 else ("**" if abs(ratio) > 2 else ("*" if abs(ratio) > 1 else ""))
                print(f"  k={k}: TAR={cb.mean():.4f} (n={len(cb)})  delta={delta:+.4f}  delta/SEM={ratio:+.2f} {sig}  (degenerate={sub[sub.Interval==k].Degenerate.mean():.1%})")


if __name__ == "__main__":
    main()
