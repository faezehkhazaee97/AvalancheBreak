import matplotlib
matplotlib.use("Agg")

# ============================================================
# MECHANISTIC ATTENTION CAPTURE, RE-RUN WITH THE PROPER CHAT TEMPLATE
# ============================================================
# Same rationale as run_core_templated.py, applied to the Figures 2-6
# mechanistic-interpretability pipeline instead of the accuracy grid. The
# original capture (attention_run_0.npz + attention_metrics.csv) fed
# Qwen2.5-7B-Instruct raw, unwrapped prompt text via tokenizer(prompt).
# This script reruns the SAME representative example (A=2, N=16, k=2,
# ATTENTION_SEED=123, space-separated characters) AND the SAME
# multi-condition sweep (lengths, intervals, trial count, seed formula)
# with ONLY the prompting method changed to tokenizer.apply_chat_template,
# saving under "_templated" filenames so the raw-prompt attention data is
# preserved unchanged for direct side-by-side comparison.
#
# Cropping to the "Input: ... Output:" token span works the same way as
# scripts/inspect_attention_raw_vs_templated.py's spot-check (which already
# verified the cropped token span is IDENTICAL between raw and templated
# for this exact prompt family, and that per-cell attention correlation
# inside that span is 0.994-0.999) -- this script just does that at full
# scale, across the whole sweep, instead of one example.

import os

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
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
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads

ATTENTION_SEED = 123
random.seed(ATTENTION_SEED)


class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]

    def generate_paired_prompts(self, length, interval, rng=None):
        rng = rng or random
        raw_chars = rng.choices(self.alphabet, k=length)

        base_str = " ".join(raw_chars)
        prompt_base = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
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
        cell_str = " ".join(cellular_w_bounds)

        prompt_cell = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
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
        input_idx = next(i for i, t in enumerate(tokens) if 'Input' in t)
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


print("\nRunning forward pass for Base Prompt (representative example, templated)...")
attn_base_tensor, tokens_base = extract_all_layers_and_heads(prompt_base)

print("Running forward pass for Cellular Prompt (representative example, templated)...")
attn_cell_tensor, tokens_cell = extract_all_layers_and_heads(prompt_cell)

print("Saving full attention tensors (templated)...")
np.savez_compressed(
    os.path.join(RESULTS_DIR, "attention_run_0_templated.npz"),
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


CLUSTERING_THRESHOLD_MULTIPLIER = 0.5

def compute_condition_clustering(attn_base_tensor, attn_cell_tensor,
                                  multiplier=CLUSTERING_THRESHOLD_MULTIPLIER):
    def clustering_of(m_avg):
        threshold = multiplier / m_avg.shape[0]
        m = m_avg.copy()
        m[m < threshold] = 0.0
        G = nx.from_numpy_array(m, create_using=nx.DiGraph)
        G.remove_edges_from(nx.selfloop_edges(G))
        vals = list(nx.clustering(G, weight='weight').values())
        return np.nanmean(vals) if vals else 0.0

    base_clustering = clustering_of(attn_base_tensor.mean(axis=(0, 1)))
    cell_clustering = clustering_of(attn_cell_tensor.mean(axis=(0, 1)))
    return base_clustering, cell_clustering


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
print(f"\nStarting TEMPLATED attention sweep: {len(sweep_conditions)} conditions "
      f"({len(sweep_conditions) * 2} forward passes)...")

sweep_rows = []
for (N, k, t) in tqdm(sweep_conditions, desc="Attention sweep (templated)"):
    local_seed = 50_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    p_base, p_cell = engine.generate_paired_prompts(length=N, interval=k, rng=rng)
    ab, _ = extract_all_layers_and_heads(p_base)
    ac, tc = extract_all_layers_and_heads(p_cell)

    m = compute_condition_metrics(ab, ac, tc)
    m["Delta_Diag"] = m["Cell_Diag"] - m["Base_Diag"]
    m["Delta_Entropy"] = m["Cell_Entropy"] - m["Base_Entropy"]
    m["Delta_Dist"] = m["Cell_Dist"] - m["Base_Dist"]
    m["Base_Clustering"], m["Cell_Clustering"] = compute_condition_clustering(ab, ac)
    m["Delta_Clustering"] = m["Cell_Clustering"] - m["Base_Clustering"]
    m.update(Seq_Length=N, Interval=k, Trial=t)
    sweep_rows.append(m)

sweep_df = pd.DataFrame(sweep_rows)
sweep_df["Inv_K"] = 1.0 / sweep_df["Interval"]

csv_filename = os.path.join(RESULTS_DIR, "attention_metrics_templated.csv")
sweep_df.to_csv(csv_filename, index=False)
print(f"\nAttention sweep complete. Saved {len(sweep_df)} condition-level rows to '{csv_filename}'")

summary = (
    sweep_df.groupby(["Seq_Length", "Interval"])[["Cell_Boundary_Attn", "Delta_Entropy", "Delta_Diag"]]
            .mean()
            .reset_index()
)
print("\n--- MEAN ATTENTION METRICS BY (Seq_Length, Interval), templated ---")
print(summary.to_string(index=False))

print("\n--- BOUNDARY ATTENTION vs 1/k REGRESSION, BY LENGTH (templated) ---")
for N, sub in sweep_df.groupby("Seq_Length"):
    if sub["Inv_K"].nunique() >= 2:
        slope, intercept, r, p, se = linregress(sub["Inv_K"], sub["Cell_Boundary_Attn"])
        print(f"N={N:>4}: slope={slope:.4f}  R^2={r**2:.4f}")

print("\nTemplated mechanistic attention capture complete.")
