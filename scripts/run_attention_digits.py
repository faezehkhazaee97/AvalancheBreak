import matplotlib
matplotlib.use("Agg")

# ============================================================
# MECHANISTIC ATTENTION CAPTURE, DIGIT ALPHABET (tokenization-clean, no
# space-separation workaround)
# ============================================================
# scripts/run_attention_templated.py (the letter-alphabet version this
# mirrors) uses SPACE-SEPARATED characters ("A A A A B A B A...") to force
# clean one-token-per-character tokenization -- necessary because letters
# merge into multi-character tokens otherwise (confirmed: Qwen's tokenizer
# contains "ABCDEFG" as a single token). That workaround means the
# attention analysis was measured on a DIFFERENT prompt format than the
# actual accuracy-grid experiments, which use concatenated, no-space
# letters.
#
# This script instead swaps the alphabet to digits (0,1 for alphabet_size=2,
# matching tasks/cyclic_shift_digits/engine.py), which tokenize one-token-
# per-character on Qwen with NO space-separation needed (verified in the
# alphabet tokenization audit: digit ratio is exactly 1.0 on every
# documented model). So this capture uses the EXACT SAME concatenated,
# no-space prompt format as the real accuracy-grid experiments -- zero
# confound, zero artificial workaround, genuinely representative of what
# the model actually sees in the real experiment.
#
# Same representative example (N=16, k=2, alphabet_size=2, ATTENTION_SEED=
# 123) and same sweep (lengths, intervals, trial count, seed formula) as
# the letter-alphabet version, for direct side-by-side comparison.

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import linregress
from tqdm import tqdm

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads

ATTENTION_SEED = 123
random.seed(ATTENTION_SEED)


class TaskEngine:
    """Digit alphabet, concatenated (no space-join) -- matches
    tasks/cyclic_shift_digits/engine.py's real prompt format exactly."""

    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [str(i) for i in range(alphabet_size)]

    def generate_paired_prompts(self, length, interval, rng=None):
        rng = rng or random
        raw_chars = rng.choices(self.alphabet, k=length)

        base_str = "".join(raw_chars)  # concatenated, NOT space-separated
        prompt_base = f"""You are a deterministic sequence transformer.
Task: Cyclic digit shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every digit with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {base_str}
Output: """

        cellular_w_bounds = []
        for i, char in enumerate(raw_chars):
            cellular_w_bounds.append(char)
            if interval > 0 and (i + 1) % interval == 0 and i != length - 1:
                cellular_w_bounds.append("|")
        cell_str = "".join(cellular_w_bounds)  # concatenated, NOT space-separated

        prompt_cell = f"""You are a deterministic sequence transformer.
Task: Cyclic digit shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every digit with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {cell_str}
Output: """
        return prompt_base, prompt_cell


engine = TaskEngine(alphabet_size=2)
N_length = 16
k_interval = 2
prompt_base, prompt_cell = engine.generate_paired_prompts(length=N_length, interval=k_interval)


def extract_all_layers_and_heads(prompt):
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]

    try:
        input_idx = max(i for i, t in enumerate(tokens) if 'Input' in t)  # crop fix, safe either way
        output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])
        start_idx = input_idx + 1
        while start_idx < output_idx and tokens[start_idx].strip() in [':', '']:
            start_idx += 1
        end_idx = output_idx
        while end_idx > start_idx and tokens[end_idx - 1].strip() == '':
            end_idx -= 1
    except StopIteration:
        start_idx, end_idx = 0, len(tokens)

    clean_tokens = [t.strip() if t.strip() else '_' for t in tokens[start_idx:end_idx]]

    attentions = outputs.attentions
    seq_len = end_idx - start_idx

    full_tensor = np.zeros((num_layers, num_heads, seq_len, seq_len), dtype=np.float32)
    for l in range(num_layers):
        layer_attn = attentions[l][0]
        for h in range(num_heads):
            attn_matrix = layer_attn[h].cpu().to(torch.float32).numpy()
            full_tensor[l, h] = attn_matrix[start_idx:end_idx, start_idx:end_idx]

    return full_tensor, clean_tokens


print("\nRunning forward pass for Base Prompt (representative example, digit alphabet)...")
attn_base_tensor, tokens_base = extract_all_layers_and_heads(prompt_base)
print(f"Base token count: {len(tokens_base)} (intended chars: {N_length}) -- "
      f"{'CLEAN 1:1' if len(tokens_base) == N_length else 'MISMATCH -- merging occurred'}")
print(f"Base tokens: {tokens_base}")

print("Running forward pass for Cellular Prompt (representative example, digit alphabet)...")
attn_cell_tensor, tokens_cell = extract_all_layers_and_heads(prompt_cell)
print(f"Cellular tokens: {tokens_cell}")

print("Saving full attention tensors (digit alphabet)...")
np.savez_compressed(
    os.path.join(RESULTS_DIR, "attention_run_0_digits.npz"),
    attn_base=attn_base_tensor,
    attn_cell=attn_cell_tensor,
    tokens_base=np.array(tokens_base),
    tokens_cell=np.array(tokens_cell),
    N_length=N_length,
    k_interval=k_interval,
)
print("Full tensors saved successfully.\n")


def compute_condition_metrics(attn_base_tensor, attn_cell_tensor, tokens_cell):
    boundary_indices = [i for i, t in enumerate(tokens_cell) if '|' in t]
    eps = 1e-12
    rows = []
    for l in range(num_layers):
        for h in range(num_heads):
            m_base = attn_base_tensor[l, h]
            m_cell = attn_cell_tensor[l, h]

            N_base = m_base.shape[0]
            N_cell = m_cell.shape[0]

            if N_base > 1:
                base_diag = np.mean(np.diag(m_base))
                base_entropy = np.mean(-np.sum(m_base * np.log2(m_base + eps), axis=-1))
                dist_base = np.abs(np.arange(N_base)[:, None] - np.arange(N_base)[None, :]) / N_base
                base_dist = np.mean(np.sum(m_base * dist_base, axis=-1))
            else:
                base_diag = base_entropy = base_dist = 0.0

            if N_cell > 1:
                cell_diag = np.mean(np.diag(m_cell))
                cell_entropy = np.mean(-np.sum(m_cell * np.log2(m_cell + eps), axis=-1))
                dist_cell = np.abs(np.arange(N_cell)[:, None] - np.arange(N_cell)[None, :]) / N_cell
                cell_dist = np.mean(np.sum(m_cell * dist_cell, axis=-1))
                if len(boundary_indices) > 0:
                    cell_boundary = np.mean(np.sum(m_cell[:, boundary_indices], axis=-1))
                else:
                    cell_boundary = 0.0
            else:
                cell_diag = cell_entropy = cell_dist = cell_boundary = 0.0

            rows.append({
                "Base_Diag": base_diag, "Cell_Diag": cell_diag,
                "Base_Entropy": base_entropy, "Cell_Entropy": cell_entropy,
                "Base_Dist": base_dist, "Cell_Dist": cell_dist,
                "Cell_Boundary_Attn": cell_boundary,
            })
    return pd.DataFrame(rows).mean().to_dict()


SWEEP_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64, 128, 256]
SWEEP_INTERVALS = [2, 4, 8]
SWEEP_TRIALS = 20

sweep_conditions = [
    (N, k, t)
    for N in SWEEP_SEQUENCE_LENGTHS
    for k in SWEEP_INTERVALS
    for t in range(SWEEP_TRIALS)
    if k < N
]
print(f"\nStarting DIGIT attention sweep: {len(sweep_conditions)} conditions "
      f"({len(sweep_conditions) * 2} forward passes)...")

sweep_rows = []
n_mismatch = 0
for (N, k, t) in tqdm(sweep_conditions, desc="Attention sweep (digits)"):
    local_seed = 50_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    p_base, p_cell = engine.generate_paired_prompts(length=N, interval=k, rng=rng)
    ab, tb = extract_all_layers_and_heads(p_base)
    ac, tc = extract_all_layers_and_heads(p_cell)
    if len(tb) != N:
        n_mismatch += 1

    m = compute_condition_metrics(ab, ac, tc)
    m["Delta_Diag"] = m["Cell_Diag"] - m["Base_Diag"]
    m["Delta_Entropy"] = m["Cell_Entropy"] - m["Base_Entropy"]
    m["Delta_Dist"] = m["Cell_Dist"] - m["Base_Dist"]
    m.update(Seq_Length=N, Interval=k, Trial=t)
    sweep_rows.append(m)

sweep_df = pd.DataFrame(sweep_rows)
sweep_df["Inv_K"] = 1.0 / sweep_df["Interval"]

csv_filename = os.path.join(RESULTS_DIR, "attention_metrics_digits.csv")
sweep_df.to_csv(csv_filename, index=False)
print(f"\nDigit attention sweep complete. Saved {len(sweep_df)} condition-level rows to '{csv_filename}'")
print(f"Tokenization mismatches (Base token count != N): {n_mismatch}/{len(sweep_conditions)}")

print("\n--- BOUNDARY ATTENTION vs 1/k REGRESSION, BY LENGTH (digits) ---")
for N, sub in sweep_df.groupby("Seq_Length"):
    if sub["Inv_K"].nunique() >= 2:
        slope, intercept, r, p, se = linregress(sub["Inv_K"], sub["Cell_Boundary_Attn"])
        print(f"N={N:>4}: slope={slope:.4f}  R^2={r**2:.4f}")

print("\nDigit-alphabet mechanistic attention capture complete.")
