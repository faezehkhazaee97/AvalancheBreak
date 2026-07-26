"""Full production-style run of reverse_xor: full grid, 40 trials/cell, on a
given model. Tracks two distinct failure modes: repetition-collapse
(is_degenerate, same heuristic used for running_xor) and GARBLED output
(non-binary/non-boundary tokens -- foreign characters, English words, code
fragments -- seen at a striking 69% rate in the 1-trial verify pass, far
higher than anything running_xor ever produced). Both get their own column
so the two failure modes don't get conflated in the significance tables.
"""

import os
import random
import re
import sys
from collections import Counter

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from tasks.common import scoring as scoring_mod
from tasks.common.runner import _build_grid, _local_seed, _seed_everything
import tasks.reverse_xor.engine as reverse_xor

TRIALS_PER_CELL = 40


def is_degenerate(s):
    s = str(s)
    if len(s) < 5:
        return False
    counts = Counter(s)
    return counts.most_common(1)[0][1] / len(s) > 0.85


def is_garbled(raw_output):
    # anything besides 0, 1, |, whitespace in the RAW (unstripped) model output
    return bool(re.search(r'[^\s01|]', str(raw_output)))


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
    grid = _build_grid(reverse_xor, trials_per_cell=TRIALS_PER_CELL)
    print(f"Grid size: {len(grid)} evaluations (full grid, {TRIALS_PER_CELL} trials/cell, model={model_tag})")

    rows = []
    for idx, (N, k, axis_val, t) in enumerate(grid):
        method = "Base" if k == 0 else "Cellular_Bounds"
        seed = _local_seed(N, k, axis_val, t)
        rng = random.Random(seed)
        input_str, truth, prompt = reverse_xor.build_prompt(N, k, rng)

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
            "Degenerate": is_degenerate(result["clean_pred"]),
            "Garbled": is_garbled(raw_output), "Raw_Model_Output": raw_output,
        })
        if idx % 300 == 0:
            print(f"  [{idx}/{len(grid)}] N={N} k={k} t={t}: TAR={result['TAR']:.3f} garbled={rows[-1]['Garbled']}")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", "reverse_xor", "verify")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{model_tag}_FULL_GRID_40trials.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}")

    print(f"\n--- TAR by Length/Interval, {model_tag}, {TRIALS_PER_CELL} trials/cell ---")
    for N in sorted(df.Seq_Length.unique()):
        sub = df[df.Seq_Length == N]
        base = sub[sub.Method == "Base"]["TAR"]
        base_tar, base_sem = base.mean(), base.std(ddof=1) / (len(base) ** 0.5)
        base_garbled = sub[sub.Method == "Base"].Garbled.mean()
        print(f"\nN={N}: Base TAR={base_tar:.4f}+/-{base_sem:.4f} (n={len(base)}), garbled={base_garbled:.1%}, degenerate={sub[sub.Method=='Base'].Degenerate.mean():.1%}")
        for k in sorted(sub[sub.Method == "Cellular_Bounds"].Interval.unique()):
            cb = sub[sub.Interval == k]["TAR"]
            cb_sem = cb.std(ddof=1) / (len(cb) ** 0.5)
            delta = cb.mean() - base_tar
            combined_sem = (cb_sem ** 2 + base_sem ** 2) ** 0.5
            ratio = delta / combined_sem if combined_sem > 0 else 0
            sig = "***" if abs(ratio) > 3 else ("**" if abs(ratio) > 2 else ("*" if abs(ratio) > 1 else ""))
            garbled_rate = sub[sub.Interval == k].Garbled.mean()
            print(f"  k={k}: TAR={cb.mean():.4f} (n={len(cb)})  delta={delta:+.4f}  delta/SEM={ratio:+.2f} {sig}  garbled={garbled_rate:.1%}")


if __name__ == "__main__":
    main()
