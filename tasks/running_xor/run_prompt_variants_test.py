"""Exploratory side-test (NOT part of the main clean pipeline): does a more
thorough prompt -- specifically, a worked example that actually contains a
boundary character, and/or an explicit verbal rule that boundaries are
"transparent" (skip over them to find the previous digit) -- help
running_xor survive boundary interleaving better than the original prompt?

The original prompt's only worked example ("Input 1011 -> Output 1110")
never shows a boundary at all, so the model never sees a concrete
demonstration of "skip the | when finding the previous digit" -- it only
gets the abstract instruction "copy boundaries unchanged, don't transform
them," which says nothing about how they interact with the XOR
calculation's notion of "previous digit."

Three variants tested as a real ablation, each as a paired comparison
against the original prompt's existing verify data (same grid, same seeds):
  - boundary_example:  add a second worked example that includes a '|'
  - explicit_skip_rule: add a verbal rule that '|' is transparent/skipped
  - both_combined:      both of the above together

Runs 3 trials/cell (not 1) specifically because the disambiguated-prompt
test and the cumulative-XOR mixup earlier this session showed how easy it
is to be fooled by n=1-per-cell noise -- more trials here to get a more
trustworthy signal before concluding anything.
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
from tasks.common.runner import _build_grid, _local_seed, _seed_everything
from tasks.common.boundary import insert_boundaries
import tasks.running_xor.engine as running_xor

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TRIALS_PER_CELL = 3

BASE_EXAMPLE = "Example: Input 1011 → Output 1110 (calculation: 1, 0⊕1=1, 1⊕0=1, 1⊕1=0)"
BOUNDARY_EXAMPLE = (
    "Example with a boundary: Input 10|11 → Output 11|10 (calculation: "
    "1, 0⊕1=1, then | is copied unchanged and skipped over, then 1⊕1=0, 1⊕1=0 "
    "-- note position 3's \"previous digit\" is position 2's digit 1, the boundary itself is never used)"
)
SKIP_RULE = (
    'IMPORTANT: boundaries (|) are transparent for this calculation -- when finding '
    '"the previous digit" for a position, skip over any \'|\' characters and use the '
    "nearest actual 0/1 digit before it. The boundary itself never participates in any "
    "XOR calculation; it is only copied to the output unchanged at the same position."
)

VARIANTS = {
    "boundary_example": lambda: [BASE_EXAMPLE, BOUNDARY_EXAMPLE],
    "explicit_skip_rule": lambda: [SKIP_RULE, BASE_EXAMPLE],
    "both_combined": lambda: [SKIP_RULE, BASE_EXAMPLE, BOUNDARY_EXAMPLE],
}


def build_prompt(N, k, rng, variant):
    input_str_base = "".join(str(rng.choice([0, 1])) for _ in range(N))
    output_str_base = running_xor._compute_xor_chain(input_str_base)
    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    extra_lines = "\n".join(VARIANTS[variant]())

    prompt = f"""You are a deterministic sequence transformer.
Task: Running XOR with previous input.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the previous input digit.
The first digit is copied as-is (no previous input to XOR with).
Each subsequent digit: output[i] = input[i] XOR input[i-1]
{extra_lines}
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """
    return input_bounded, output_bounded, prompt


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else None
    if variant not in VARIANTS:
        print(f"Usage: python run_prompt_variants_test.py <variant>. Options: {list(VARIANTS)}")
        sys.exit(1)

    print(f"Loading {MODEL_ID}... (variant={variant})")
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
        input_str, truth, prompt = build_prompt(N, k, rng, variant)

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
            "Variant": variant, "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
            "Input_Given": input_str, "Ground_Truth": result["clean_truth"],
            "LLM_Output": result["clean_pred"], "TAR": result["TAR"], "SAR": result["SAR"],
            "Raw_Model_Output": raw_output,
        })

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", "running_xor", "verify")
    out_csv = os.path.join(out_dir, f"Qwen2.5-7B-Instruct_prompt_variant_{variant}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}")
    print(f"\n--- TAR by Method ({variant}) ---")
    print(df.groupby("Method")["TAR"].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
