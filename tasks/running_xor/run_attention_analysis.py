"""Mechanistic side-analysis: does running_xor show a STRONGER attention-sink
effect from boundary tokens than cyclic_shift does -- i.e., is there direct
evidence boundaries disrupt this model's attention more for the task where
they hurt than for the task where they help?

Reuses the exact metric definitions from the paper's own proven mechanistic
pipeline (scripts/run_attention_templated.py): Cell_Boundary_Attn (mean
attention mass paid to boundary '|' tokens), Delta_Entropy (entropy
increase from Base to Cellular), Delta_Diag (self-attention change). Same
representative example convention (N=16, k=2) for direct comparability with
the paper's existing attention figures.

Two parts:
  1. Static comparison: cyclic_shift (boundaries help) vs running_xor
     (boundaries hurt), same N/k, paired Base/Cellular prompts, same metrics.
  2. Failure-point analysis: teacher-force the model's OWN real observed
     collapse continuation (N=8, k=1, Input=0|0|0|1|1|0|0|1, raw output
     't|1|1|s|1|t|1' from the original verify run) and inspect attention
     specifically at the position where it first emits an invalid character,
     compared to an earlier, still-healthy position in the same sequence.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import tasks.cyclic_shift.engine as cyclic_shift
import tasks.running_xor.engine as running_xor
from tasks.common.runner import _local_seed
import random

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def extract_attention(tokenizer, model, num_layers, num_heads, prompt, forced_continuation=None):
    """forced_continuation, if given, is appended (teacher-forced) after the
    prompt so we can inspect attention over the model's OWN real output."""
    full_text = prompt + (forced_continuation or "")
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    if forced_continuation:
        cont_ids = tokenizer(forced_continuation, add_special_tokens=False, return_tensors="pt")["input_ids"]
        input_ids = torch.cat([enc["input_ids"], cont_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        enc = {"input_ids": input_ids, "attention_mask": attention_mask}
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]

    try:
        # LAST occurrence of "Input", not first -- prompts with a worked example
        # ("Example: Input 1011 -> Output 1110...") contain an earlier, spurious
        # "Input" that the real task's "Input: {actual}" line always follows.
        input_idx = max(i for i, t in enumerate(tokens) if "Input" in t)
        output_idx = next(i for i in range(input_idx, len(tokens)) if "Output" in tokens[i])
        start_idx = input_idx + 1
        while start_idx < output_idx and tokens[start_idx].strip() in [":", ""]:
            start_idx += 1
        end_idx = len(tokens) if forced_continuation else output_idx
    except (StopIteration, ValueError):
        start_idx, end_idx = 0, len(tokens)

    clean_tokens = [t.strip() if t.strip() else "_" for t in tokens[start_idx:end_idx]]
    attentions = outputs.attentions
    seq_len = end_idx - start_idx

    full_tensor = np.zeros((num_layers, num_heads, seq_len, seq_len), dtype=np.float32)
    for l in range(num_layers):
        layer_attn = attentions[l][0]
        for h in range(num_heads):
            attn_matrix = layer_attn[h].cpu().to(torch.float32).numpy()
            full_tensor[l, h] = attn_matrix[start_idx:end_idx, start_idx:end_idx]

    return full_tensor, clean_tokens


def compute_metrics(attn_base, attn_cell, tokens_cell):
    boundary_indices = [i for i, t in enumerate(tokens_cell) if "|" in t]
    eps = 1e-12

    def stats(m):
        n = m.shape[0]
        if n <= 1:
            return 0.0, 0.0
        diag = np.mean(np.diag(m))
        entropy = np.mean(-np.sum(m * np.log2(m + eps), axis=-1))
        return diag, entropy

    base_diag, base_entropy = stats(attn_base.mean(axis=(0, 1)))
    cell_diag, cell_entropy = stats(attn_cell.mean(axis=(0, 1)))
    m_cell = attn_cell.mean(axis=(0, 1))
    cell_boundary = np.mean(np.sum(m_cell[:, boundary_indices], axis=-1)) if boundary_indices else 0.0

    return {
        "Base_Diag": base_diag, "Cell_Diag": cell_diag, "Delta_Diag": cell_diag - base_diag,
        "Base_Entropy": base_entropy, "Cell_Entropy": cell_entropy, "Delta_Entropy": cell_entropy - base_entropy,
        "Cell_Boundary_Attn": cell_boundary,
    }


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # ===== Part 1: static comparison, same N/k, cyclic_shift vs running_xor =====
    N, k = 16, 2
    print(f"\n=== Part 1: static attention comparison, N={N}, k={k} ===")

    rng_base = random.Random(_local_seed(N, 0, 2, 0))
    cs_base_in, _, cs_base_prompt = cyclic_shift.build_prompt(N, 0, rng_base, axis_val=2)
    rng_cell = random.Random(_local_seed(N, k, 2, 0))
    cs_cell_in, _, cs_cell_prompt = cyclic_shift.build_prompt(N, k, rng_cell, axis_val=2)

    rng_base2 = random.Random(_local_seed(N, 0, None, 0))
    xor_base_in, _, xor_base_prompt = running_xor.build_prompt(N, 0, rng_base2)
    rng_cell2 = random.Random(_local_seed(N, k, None, 0))
    xor_cell_in, _, xor_cell_prompt = running_xor.build_prompt(N, k, rng_cell2)

    print("Extracting: cyclic_shift Base...")
    cs_ab, _ = extract_attention(tokenizer, model, num_layers, num_heads, cs_base_prompt)
    print("Extracting: cyclic_shift Cellular...")
    cs_ac, cs_tc = extract_attention(tokenizer, model, num_layers, num_heads, cs_cell_prompt)
    print("Extracting: running_xor Base...")
    xor_ab, _ = extract_attention(tokenizer, model, num_layers, num_heads, xor_base_prompt)
    print("Extracting: running_xor Cellular...")
    xor_ac, xor_tc = extract_attention(tokenizer, model, num_layers, num_heads, xor_cell_prompt)

    cs_metrics = compute_metrics(cs_ab, cs_ac, cs_tc)
    xor_metrics = compute_metrics(xor_ab, xor_ac, xor_tc)

    comparison = pd.DataFrame([
        {"Task": "cyclic_shift (boundaries HELP)", **cs_metrics},
        {"Task": "running_xor (boundaries HURT)", **xor_metrics},
    ])
    print("\n--- Static attention metrics: cyclic_shift vs running_xor, same N/k ---")
    print(comparison.round(4).to_string(index=False))

    out_dir = os.path.join(REPO_ROOT, "results_clean", "running_xor")
    os.makedirs(out_dir, exist_ok=True)
    comparison.to_csv(os.path.join(out_dir, "attention_comparison_cyclic_shift_vs_xor.csv"), index=False)

    # ===== Part 2: attention AT the failure point, using a real observed collapse =====
    print(f"\n=== Part 2: attention at the actual moment of collapse (N=8, k=1) ===")
    N2, k2, t2 = 8, 1, 0
    rng_fail = random.Random(_local_seed(N2, k2, None, t2))
    fail_in, fail_truth, fail_prompt = running_xor.build_prompt(N2, k2, rng_fail)
    print(f"Input: {fail_in}  (expected: {fail_truth})")

    # the real observed collapse continuation from the original verify run
    real_collapse_output = "t|1|1|s|1|t|1"
    print(f"Teacher-forcing the model's OWN real observed output: {real_collapse_output!r}")

    attn_full, tokens_full = extract_attention(tokenizer, model, num_layers, num_heads, fail_prompt, forced_continuation=real_collapse_output)
    print("Tokens (prompt input span + forced continuation):", tokens_full)

    # find the position of the first invalid character ('t' or 's') in the continuation
    fail_token_idx = next((i for i, t in enumerate(tokens_full) if t in ("t", "s")), None)
    healthy_token_idx = next((i for i, t in enumerate(tokens_full) if t in ("0", "1")), None)

    if fail_token_idx is not None and healthy_token_idx is not None:
        avg_attn = attn_full.mean(axis=(0, 1))  # avg over layers/heads: (seq, seq)
        boundary_idx = [i for i, t in enumerate(tokens_full) if "|" in t]

        for label, idx in [("healthy position (early '1')", healthy_token_idx),
                            ("FAILURE position (first 't'/'s')", fail_token_idx)]:
            row = avg_attn[idx]
            attn_to_boundaries = row[boundary_idx].sum() if boundary_idx else 0.0
            entropy = -np.sum(row * np.log2(row + 1e-12))
            print(f"\n{label} (token idx {idx}):")
            print(f"  attention mass to boundary '|' tokens: {attn_to_boundaries:.4f}")
            print(f"  attention entropy: {entropy:.4f}")
    else:
        print("Could not locate both a healthy and a failure position in the tokenized continuation -- tokenization may have split characters differently than expected.")

    print("\nDone.")


if __name__ == "__main__":
    main()
