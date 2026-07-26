"""Exploratory side-test (NOT part of the main clean pipeline): does making
the prompt explicitly warn against the cumulative/parity-XOR
misinterpretation (XOR-ing against the model's own previous OUTPUT instead
of the previous raw INPUT digit) reduce that specific confusion?

Motivated by a direct data finding: in 8/24 (33%) of Cellular_Bounds verify
trials, the model's actual wrong answer matched a "cumulative XOR" (XOR
against the running output, like a parity accumulator) better than the
correct rule (XOR against the previous raw input digit) -- including at
least one exact match (Input=1001: correct=1101, model said 1110, which is
exactly the cumulative-XOR result). The original prompt's formula is
technically unambiguous, but "running XOR" is a much more common pattern
in general training data as a cumulative accumulator, so the model may
default toward it under difficulty even when the stated formula says
otherwise.

Reuses the exact same grid and seeds as
results_clean/running_xor/verify/Qwen2.5-7B-Instruct.csv for a clean paired
comparison; only the prompt wording differs (task logic/scoring unchanged).
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


def build_disambiguated_prompt(N, k, rng):
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
IMPORTANT: input[i-1] means the ORIGINAL INPUT digit at the previous position, read directly from the Input line -- NOT a digit you have already written in your Output. This is NOT a cumulative or running XOR of your own outputs; each output digit depends only on the two original input digits at that position and the position before it, never on any digit you have already output.
Example: Input 1011 → Output 1110 (calculation: 1, 0⊕1=1, 1⊕0=1, 1⊕1=0)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """
    return input_bounded, output_bounded, prompt


def cumulative_xor_chain(inp):
    out = [inp[0]]
    for i in range(1, len(inp)):
        out.append(str(int(inp[i]) ^ int(out[i - 1])))
    return "".join(out)


def tar_against(pred, truth):
    n = len(truth)
    matches = sum(1 for i in range(n) if i < len(pred) and pred[i] == truth[i])
    return matches / n if n else 0.0


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    _seed_everything()
    grid = _build_grid(running_xor, trials_per_cell=1)
    print(f"Grid size: {len(grid)} evaluations (matches the existing verify pass)")

    rows = []
    for N, k, axis_val, t in grid:
        method = "Base" if k == 0 else "Cellular_Bounds"
        seed = _local_seed(N, k, axis_val, t)
        rng = random.Random(seed)
        input_str, truth, prompt = build_disambiguated_prompt(N, k, rng)

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

        plain_in = input_str.replace("|", "")
        alt_truth = cumulative_xor_chain(plain_in)
        tar_alt = tar_against(result["clean_pred"], alt_truth)

        rows.append({
            "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
            "Input_Given": input_str, "Ground_Truth": result["clean_truth"],
            "LLM_Output": result["clean_pred"], "TAR": result["TAR"], "SAR": result["SAR"],
            "TAR_vs_cumulative_alt": tar_alt, "alt_fits_better": tar_alt > result["TAR"],
            "Raw_Model_Output": raw_output,
        })
        print(f"  N={N} k={k} ({method}): TAR={result['TAR']:.3f} TAR_alt={tar_alt:.3f}")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", "running_xor", "verify")
    out_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_disambiguated_prompt_test.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}")

    print("\n--- TAR by Method: disambiguated prompt ---")
    print(df.groupby("Method")["TAR"].mean())

    cb = df[df.Method == "Cellular_Bounds"]
    print(f"\nCellular_Bounds rows where cumulative-XOR still fits better: {cb['alt_fits_better'].sum()} / {len(cb)}")
    print(f"Mean TAR vs correct (disambiguated prompt, Cellular_Bounds only): {cb['TAR'].mean():.4f}")
    print(f"Mean TAR vs cumulative alt (disambiguated prompt, Cellular_Bounds only): {cb['TAR_vs_cumulative_alt'].mean():.4f}")

    # --- compare against the original (ambiguous) prompt's existing verify data ---
    orig = pd.read_csv(os.path.join(out_dir, "Qwen2.5-7B-Instruct.csv"))
    merged = orig.merge(df, on=["Method", "Interval", "Seq_Length", "Trial"], suffixes=("_original", "_disambiguated"))
    merged["TAR_delta"] = merged["TAR_disambiguated"] - merged["TAR_original"]
    print("\n--- TAR delta (disambiguated - original), by Method ---")
    print(merged.groupby("Method")["TAR_delta"].mean())
    comp_csv = os.path.join(out_dir, "Qwen2.5-7B-Instruct_disambiguated_prompt_comparison.csv")
    merged[["Method", "Interval", "Seq_Length", "TAR_original", "TAR_disambiguated", "TAR_delta"]].to_csv(comp_csv, index=False)
    print(f"\nSaved comparison to {comp_csv}")


if __name__ == "__main__":
    main()
