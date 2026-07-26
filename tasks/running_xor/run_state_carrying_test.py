"""Exploratory side-test (NOT part of the main clean pipeline): does
replacing running_xor's content-free "|" boundary with a checkpoint that
states the TRUE running XOR state at that point help, the way the
already-validated modsum state-carrying redesign (scripts/run_modsum_
state_carrying.py) did for that task?

XOR is the cleanest possible case for this idea: its entire "state" is a
single bit (the most recent output value), so the checkpoint just needs to
say that one bit -- no abstract running total to communicate.

Reuses the exact same grid and per-cell seeds as
results_clean/running_xor/verify/Qwen2.5-7B-Instruct.csv, so Base and the
original content-free Cellular_Bounds rows from that file are reused
directly (not regenerated) for a clean three-way paired comparison; only
the new Cellular_StateCarrying condition is run fresh here.
"""

import os
import random
import re
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from tasks.common.boundary import valid_intervals, STANDARD_LENGTHS, STANDARD_INTERVALS
from tasks.common.runner import _local_seed, _seed_everything
from tasks.running_xor.engine import _compute_xor_chain

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def build_state_carrying_prompt(N, k, rng):
    input_str_base = "".join(str(rng.choice([0, 1])) for _ in range(N))
    output_str_base = _compute_xor_chain(input_str_base)

    input_w_bounds, output_w_bounds = "", ""
    for i, (d, out_d) in enumerate(zip(input_str_base, output_str_base)):
        input_w_bounds += d
        output_w_bounds += out_d
        if k > 0 and (i + 1) % k == 0 and i != N - 1:
            checkpoint = f"[{out_d}]"
            input_w_bounds += checkpoint
            output_w_bounds += checkpoint

    prompt = f"""You are a deterministic sequence transformer.
Task: Running XOR with previous input.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the previous input digit.
The first digit is copied as-is (no previous input to XOR with).
Each subsequent digit: output[i] = input[i] XOR input[i-1]
Constraint: The input contains bracketed checkpoints like [{output_str_base[0]}]. Each checkpoint states the TRUE, correct XOR-chain output at that exact point -- i.e. the most recent output bit. Use it directly as your "previous digit" for continuing the chain going forward -- you do not need to re-derive it from the digits before it. Copy each checkpoint exactly to the output, unchanged, at the same position.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
    return input_w_bounds, output_w_bounds, prompt


def score_state_carrying(raw_output, clean_truth):
    pred = raw_output.strip().split("Explanation")[0].split("\n")[0].strip()
    clean_pred = re.sub(r"\[\d\]", "", pred).replace(" ", "").replace("\n", "")
    clean_pred = clean_pred[: len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0
    return clean_pred, tar, sar


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    _seed_everything()
    grid = [(N, k, 0) for N in STANDARD_LENGTHS for k in valid_intervals(N, STANDARD_INTERVALS) if k > 0]
    print(f"Grid size: {len(grid)} evaluations (Cellular conditions only -- Base is reused unchanged from the existing verify CSV)")

    rows = []
    for N, k, t in grid:
        seed = _local_seed(N, k, None, t)
        rng = random.Random(seed)
        input_str, truth, prompt = build_state_carrying_prompt(N, k, rng)
        clean_truth = re.sub(r"\[\d\]", "", truth)

        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}
        max_new = int(N * 2.0) + 20  # checkpoints add extra output tokens, same margin as modsum's version

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_only = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)
        clean_pred, tar, sar = score_state_carrying(raw_output, clean_truth)

        rows.append({
            "Method": "Cellular_StateCarrying", "Interval": k, "Seq_Length": N, "Trial": t,
            "Input_Given": input_str, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
            "TAR": tar, "SAR": sar, "Raw_Model_Output": raw_output,
        })
        print(f"  N={N} k={k}: raw={raw_output!r} TAR={tar:.3f}")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", "running_xor", "verify")
    os.makedirs(out_dir, exist_ok=True)
    state_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_state_carrying_test.csv")
    df.to_csv(state_csv, index=False)
    print(f"\nSaved {len(df)} rows to {state_csv}")

    # --- three-way comparison against the existing verify CSV (Base + content-free Cellular_Bounds) ---
    existing = pd.read_csv(os.path.join(out_dir, "Qwen2.5-7B-Instruct.csv"))
    base_ref = existing[existing.Method == "Base"].groupby("Seq_Length")["TAR"].mean()
    comp_rows = []
    for N in sorted(df.Seq_Length.unique()):
        for k in sorted(df[df.Seq_Length == N].Interval.unique()):
            old_cell = existing[(existing.Method == "Cellular_Bounds") & (existing.Seq_Length == N) & (existing.Interval == k)]["TAR"]
            new_cell = df[(df.Seq_Length == N) & (df.Interval == k)]["TAR"]
            comp_rows.append({
                "Seq_Length": N, "Interval": k,
                "Base_TAR": base_ref.get(N, float("nan")),
                "Cellular_ContentFree_TAR": old_cell.mean() if len(old_cell) else float("nan"),
                "Cellular_StateCarrying_TAR": new_cell.mean() if len(new_cell) else float("nan"),
            })
    comparison = pd.DataFrame(comp_rows)
    comp_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_state_carrying_comparison.csv")
    comparison.to_csv(comp_csv, index=False)
    print(f"\n--- DOES A STATE-CARRYING BOUNDARY HELP WHERE A CONTENT-FREE ONE DOESN'T? ---")
    print(comparison.round(4).to_string(index=False))
    print(f"\nSaved comparison to {comp_csv}")


if __name__ == "__main__":
    main()
