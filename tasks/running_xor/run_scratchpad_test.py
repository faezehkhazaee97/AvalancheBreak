"""Exploratory side-test (NOT part of the main clean pipeline): does letting
the model work through the XOR chain step-by-step (a scratchpad), instead
of forcing it to compute and emit the final digit stream simultaneously,
help under boundary interleaving?

This is a fundamentally different lever than the prompt-clarity variants
already tested (which all failed, with tight error bars) -- those kept the
same "answer only, no intermediate work" format and just added more
explanation. This changes what the model is *allowed to do*: reason
step-by-step first, THEN commit to a final compact answer, which is the
only part that gets scored (the reasoning trace is parsed out and
discarded, matching the same scoring convention as every other task).

Note: this needs a much larger token budget than the other tests (the
model has to write out N step-by-step lines, not just N compact output
characters), so it's slower per trial. Starting with TRIALS_PER_CELL=2 as a
first look given the unknown cost profile.
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

from tasks.common.runner import _build_grid, _local_seed, _seed_everything
from tasks.common.boundary import insert_boundaries
import tasks.running_xor.engine as running_xor

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TRIALS_PER_CELL = 2

FEWSHOT_EXAMPLE = """Example:
Input: 1011
Step-by-step:
pos0: input=1, no previous digit, output=1
pos1: input=0, previous=1, 0⊕1=1
pos2: input=1, previous=0, 1⊕0=1
pos3: input=1, previous=1, 1⊕1=0
Answer: 1110"""


def build_scratchpad_prompt(N, k, rng):
    input_str_base = "".join(str(rng.choice([0, 1])) for _ in range(N))
    output_str_base = running_xor._compute_xor_chain(input_str_base)
    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running XOR with previous input.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the previous input digit.
The first digit is copied as-is (no previous input to XOR with).
Each subsequent digit: output[i] = input[i] XOR input[i-1]
Constraint: The input may contain vertical bars (|) as structural boundaries. Skip over them when finding "the previous digit," and copy them unchanged into your final answer at the same position -- they are never used in any XOR calculation.

Work through the calculation step by step, one position at a time (use the ORIGINAL INPUT digits for pos/previous, not digits from your own output). Then give the final answer, with boundaries included, on its own line starting with "Answer: ".

{FEWSHOT_EXAMPLE}

Now solve this one.
Input: {input_bounded}
Step-by-step:
"""
    return input_bounded, output_bounded, prompt


def parse_scratchpad_answer(raw_output, clean_truth):
    match = re.search(r"Answer:\s*([01|]+)", raw_output)
    pred = match.group(1) if match else ""
    clean_pred = pred.replace("|", "")[: len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0
    found_answer_marker = match is not None
    return clean_pred, tar, sar, found_answer_marker


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    _seed_everything()
    grid = _build_grid(running_xor, trials_per_cell=TRIALS_PER_CELL)
    print(f"Grid size: {len(grid)} evaluations ({TRIALS_PER_CELL} trials/cell)")

    rows = []
    for N, k, axis_val, t in grid:
        method = "Base" if k == 0 else "Cellular_Bounds"
        seed = _local_seed(N, k, axis_val, t)
        rng = random.Random(seed)
        input_str, truth, prompt = build_scratchpad_prompt(N, k, rng)
        clean_truth = truth.replace("|", "")

        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}
        max_new = N * 20 + 60  # much larger budget: step-by-step trace + final answer line

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_only = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)
        clean_pred, tar, sar, found_marker = parse_scratchpad_answer(raw_output, clean_truth)

        rows.append({
            "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
            "Input_Given": input_str, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
            "TAR": tar, "SAR": sar, "Found_Answer_Marker": found_marker,
            "Raw_Model_Output": raw_output,
        })
        print(f"  N={N} k={k} ({method}) t={t}: TAR={tar:.3f} marker_found={found_marker}")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", "running_xor", "verify")
    out_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_scratchpad_test.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}")
    print(f"\nRows where 'Answer:' marker was found: {df['Found_Answer_Marker'].sum()} / {len(df)}")
    print("\n--- TAR by Method (scratchpad) ---")
    print(df.groupby("Method")["TAR"].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
