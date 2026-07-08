import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ============================================================
# MULTI-TRIAL VERSION OF THE ILLUSTRATIVE ATTENTION FIGURES
# ============================================================
# Figures 10-16 were all built from a SINGLE example (one random draw at
# alphabet=2, N=16, k=2) -- matching how the original draft's own Figures
# 2-6 were built, but checked against the real 360-trial statistical
# sweep, at least one metric (entropy) came out reversed on that single
# draw. This reruns the SAME configuration 20 times (20 different actual
# random sequences, same alphabet/length/interval), captures the FULL
# per-layer/per-head attention tensor every time, and rebuilds every
# figure from real averages/pooled data instead of one draw. Kept small
# (N=16-23) deliberately -- this is NOT the full N-up-to-256 sweep, which
# would be storage-prohibitive at full tensor resolution; it's just many
# more draws of the one small illustrative condition already used for the
# figures going into the paper.

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads


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
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}
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
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {cell_str}
Output: """
        return prompt_base, prompt_cell


def extract_all_layers_and_heads(prompt):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0]]

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


engine = TaskEngine(alphabet_size=2)
N_length, k_interval = 16, 2
N_TRIALS = 20

sum_base, sum_cell = None, None
entropy_base_trials, entropy_cell_trials = [], []
offdiag_base_vals, offdiag_cell_vals = [], []
offdiag_base_mass, offdiag_cell_mass = [], []
distdecay_rows = []
boundary_attn_trials = []
tokens_base_ref, tokens_cell_ref = None, None

def head_entropy(attn_tensor):
    # Matches the sweep's own formula exactly (cell 37, compute_condition_metrics):
    # entropy computed directly on the raw cropped weights, NOT row-normalized first.
    # Rows don't sum to 1 within the crop (see the reading note on Fig 10), so this
    # differs from a textbook normalized-distribution entropy -- but it's what the
    # comparison sweep data uses, so it has to match for an apples-to-apples check.
    eps = 1e-12
    L, H = attn_tensor.shape[:2]
    ent = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            m = attn_tensor[l, h]
            ent[l, h] = np.mean(-np.sum(m * np.log2(m + eps), axis=-1))
    return ent

def offdiag_values(attn_tensor):
    L, H = attn_tensor.shape[:2]
    vals = []
    for l in range(L):
        for h in range(H):
            m = attn_tensor[l, h]
            N = m.shape[0]
            for i in range(N):
                vals.extend(np.delete(m[i, :], i))
    return np.array(vals)

def offdiag_mass_per_row(attn_tensor):
    L, H = attn_tensor.shape[:2]
    vals = []
    for l in range(L):
        for h in range(H):
            m = attn_tensor[l, h]
            off = np.tril(m, k=-1).sum(axis=1)
            vals.extend(off[1:])
    return np.array(vals)

print(f"\nRunning {N_TRIALS} trials at Alphabet=2, N={N_length}, k={k_interval}...")
for t in range(N_TRIALS):
    rng = random.Random(9000 + t * 137)
    prompt_base, prompt_cell = engine.generate_paired_prompts(length=N_length, interval=k_interval, rng=rng)

    attn_base, tokens_base = extract_all_layers_and_heads(prompt_base)
    attn_cell, tokens_cell = extract_all_layers_and_heads(prompt_cell)

    if sum_base is None:
        sum_base = np.zeros_like(attn_base)
        sum_cell = np.zeros_like(attn_cell)
        tokens_base_ref, tokens_cell_ref = tokens_base, tokens_cell
    sum_base += attn_base
    sum_cell += attn_cell

    mask_cell = np.array([tk != '|' for tk in tokens_cell])
    attn_cell_masked = attn_cell[:, :, mask_cell, :][:, :, :, mask_cell].copy()

    entropy_base_trials.append(head_entropy(attn_base))
    entropy_cell_trials.append(head_entropy(attn_cell))  # unmasked (boundaries included) -- matches the sweep's own compute_condition_metrics exactly

    offdiag_base_vals.append(offdiag_values(attn_base))
    offdiag_cell_vals.append(offdiag_values(attn_cell_masked))
    offdiag_base_mass.append(offdiag_mass_per_row(attn_base))
    offdiag_cell_mass.append(offdiag_mass_per_row(attn_cell_masked))

    boundary_idx = np.where(np.array(tokens_cell) == '|')[0]
    per_layer_head = np.zeros((num_layers, num_heads))
    for l in range(num_layers):
        for h in range(num_heads):
            per_layer_head[l, h] = np.mean(np.sum(attn_cell[l, h][:, boundary_idx], axis=-1))
    boundary_attn_trials.append(per_layer_head)

    attn_cell_avg_trial = attn_cell.mean(axis=(0, 1))
    div_idx = np.where(np.array(tokens_cell) == "|")[0]
    alpha_idx = np.where(np.array(tokens_cell) != "|")[0]
    N_cell = attn_cell_avg_trial.shape[0]
    for i in range(1, N_cell):
        for j in range(i):
            dist = i - j
            val = attn_cell_avg_trial[i, j]
            distdecay_rows.append({"Distance": dist, "Attention": val, "Type": "Cellular (Global)", "Trial": t})
            if j in div_idx:
                distdecay_rows.append({"Distance": dist, "Attention": val, "Type": "Cellular (To Divider)", "Trial": t})
            if i in alpha_idx and j in alpha_idx:
                distdecay_rows.append({"Distance": dist, "Attention": val, "Type": "Cellular (Alpha-to-Alpha)", "Trial": t})

    attn_base_avg_trial = attn_base.mean(axis=(0, 1))
    N_base = attn_base_avg_trial.shape[0]
    for i in range(1, N_base):
        for j in range(i):
            distdecay_rows.append({"Distance": i - j, "Attention": attn_base_avg_trial[i, j], "Type": "Base (No Dividers)", "Trial": t})

    print(f"  trial {t+1}/{N_TRIALS} done")

avg_base = sum_base / N_TRIALS
avg_cell = sum_cell / N_TRIALS
mean_entropy_base = np.mean(entropy_base_trials, axis=0)
mean_entropy_cell = np.mean(entropy_cell_trials, axis=0)
pooled_offdiag_base_vals = np.concatenate(offdiag_base_vals)
pooled_offdiag_cell_vals = np.concatenate(offdiag_cell_vals)
pooled_offdiag_base_mass = np.concatenate(offdiag_base_mass)
pooled_offdiag_cell_mass = np.concatenate(offdiag_cell_mass)
mean_boundary_attn = np.mean(boundary_attn_trials, axis=0)
distdecay_df = pd.DataFrame(distdecay_rows)

print(f"\nMean entropy Base: {mean_entropy_base.mean():.4f}, Cellular: {mean_entropy_cell.mean():.4f}")
print(f"Cellular higher in {(mean_entropy_cell.mean(axis=1) > mean_entropy_base.mean(axis=1)).sum()}/{num_layers} layers (averaged over {N_TRIALS} trials)")
print(f"Mean off-diag context mass -- Base: {pooled_offdiag_base_mass.mean():.4f}, Cellular: {pooled_offdiag_cell_mass.mean():.4f}")

sns.set_theme(style="whitegrid", font_scale=1.1)

# ---- Fig 10/11 style: averaged example attention matrix, Layer 13 Head 10 ----
layer_idx, head_idx = 13, 10
mask_cell_ref = np.array([tk != '|' for tk in tokens_cell_ref])
avg_cell_masked = avg_cell[:, :, mask_cell_ref, :][:, :, :, mask_cell_ref]
m_base_avg = avg_base[layer_idx, head_idx]
m_cell_avg_masked = avg_cell_masked[layer_idx, head_idx]
m_cell_avg_full = avg_cell[layer_idx, head_idx]
pos_labels_base = [str(i) for i in range(m_base_avg.shape[0])]
pos_labels_cell_masked = [str(i) for i in range(m_cell_avg_masked.shape[0])]
pos_labels_cell_full = [tk if tk == '|' else str(i) for i, tk in enumerate(tokens_cell_ref)]
vmax_shared = max(m_base_avg.max(), m_cell_avg_masked.max())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
sns.heatmap(m_base_avg, xticklabels=pos_labels_base, yticklabels=pos_labels_base, cmap="rocket", vmin=0, vmax=vmax_shared, ax=ax1)
ax1.set_title(f"Base Attention (avg of {N_TRIALS} trials)\nLayer {layer_idx}, Head {head_idx}", fontsize=12)
ax1.set_aspect('equal')
sns.heatmap(m_cell_avg_masked, xticklabels=pos_labels_cell_masked, yticklabels=pos_labels_cell_masked, cmap="rocket", vmin=0, vmax=vmax_shared, ax=ax2)
ax2.set_title(f"Cellular Attention, boundaries masked (avg of {N_TRIALS} trials)\nLayer {layer_idx}, Head {head_idx}", fontsize=12)
ax2.set_aspect('equal')
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "example_attention_matrix_multitrial.png")
plt.savefig(out, dpi=300); plt.close(fig)
print(f"Saved: {out}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
sns.heatmap(m_base_avg, xticklabels=pos_labels_base, yticklabels=pos_labels_base, cmap="rocket", vmin=0, vmax=vmax_shared, ax=ax1)
ax1.set_title(f"Base Attention (avg of {N_TRIALS} trials)\nLayer {layer_idx}, Head {head_idx}", fontsize=12)
ax1.set_aspect('equal')
sns.heatmap(m_cell_avg_full, xticklabels=pos_labels_cell_full, yticklabels=pos_labels_cell_full, cmap="rocket", vmin=0, vmax=vmax_shared, ax=ax2)
ax2.set_title(f"Cellular Attention, boundaries included (avg of {N_TRIALS} trials)\nLayer {layer_idx}, Head {head_idx}", fontsize=12)
ax2.set_aspect('equal')
boundary_idx_ref = np.where(np.array(tokens_cell_ref) == '|')[0]
for b in boundary_idx_ref:
    ax2.axvspan(b, b + 1, color='cyan', alpha=0.15, zorder=0)
    ax2.axhspan(b, b + 1, color='cyan', alpha=0.15, zorder=0)
    ax2.axvline(x=b, color='cyan', linestyle='--', linewidth=1, alpha=0.8)
    ax2.axvline(x=b + 1, color='cyan', linestyle='--', linewidth=1, alpha=0.8)
    ax2.axhline(y=b, color='cyan', linestyle='--', linewidth=1, alpha=0.8)
    ax2.axhline(y=b + 1, color='cyan', linestyle='--', linewidth=1, alpha=0.8)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "example_attention_matrix_with_boundaries_multitrial.png")
plt.savefig(out, dpi=300); plt.close(fig)
print(f"Saved: {out}")

# ---- Fig 12 style: boundary sink, averaged ----
per_layer_mean = mean_boundary_attn.mean(axis=1)
peak_layer = int(np.argmax(per_layer_mean))
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.plot(range(num_layers), per_layer_mean, marker='o', markersize=5, color='#2b6cb0')
ax.scatter([peak_layer], [per_layer_mean[peak_layer]], color='crimson', zorder=5, s=60)
ax.annotate(f'Spike at Layer {peak_layer}', xy=(peak_layer, per_layer_mean[peak_layer]),
            xytext=(peak_layer, per_layer_mean[peak_layer] + 0.02 * max(per_layer_mean)),
            ha='center', color='crimson', fontsize=10, fontweight='bold')
ax.set_xlabel("Model Layer Depth"); ax.set_ylabel("Mean Boundary Attention Mass")
ax.set_title(f"Boundary Attention per Layer (avg of {N_TRIALS} trials)", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax = axes[1]
sns.heatmap(mean_boundary_attn.T, cmap="rocket", ax=ax, cbar_kws={'label': 'Attention Mass -> Boundaries'})
ax.set_xlabel("Model Layer Depth"); ax.set_ylabel("Attention Head Index")
ax.set_title(f"Boundary Attention Heatmap (avg of {N_TRIALS} trials)", fontsize=12, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "boundary_sink_analysis_multitrial.png")
plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
print(f"Saved: {out} (peak layer {peak_layer}, {per_layer_mean[peak_layer]*100:.1f}%)")

# ---- Fig 13/14 style: entropy, averaged ----
layer_mean_base_e = mean_entropy_base.mean(axis=1)
layer_mean_cell_e = mean_entropy_cell.mean(axis=1)
fig = plt.figure(figsize=(8, 5))
plt.plot(layer_mean_base_e, label="Base")
plt.plot(layer_mean_cell_e, label="Cellular")
plt.plot(layer_mean_cell_e - layer_mean_base_e, label="Delta (Cell-Base)", linestyle="--")
plt.xlabel("Layer"); plt.ylabel("Mean Head Entropy")
plt.title(f"Layer-wise Mean Attention Entropy (avg of {N_TRIALS} trials)")
plt.legend(); plt.tight_layout()
out = os.path.join(RESULTS_DIR, "head_entropy_layerwise_multitrial.png")
plt.savefig(out, dpi=300); plt.close()
print(f"Saved: {out}")

fig = plt.figure(figsize=(8, 5))
plt.hist(mean_entropy_base.flatten(), bins=40, alpha=0.6, label="Base")
plt.hist(mean_entropy_cell.flatten(), bins=40, alpha=0.6, label="Cellular")
plt.xlabel("Head Entropy"); plt.ylabel("Frequency")
plt.title(f"Distribution of Head Entropy (avg of {N_TRIALS} trials)")
plt.legend(); plt.tight_layout()
out = os.path.join(RESULTS_DIR, "head_entropy_distribution_multitrial.png")
plt.savefig(out, dpi=300); plt.close()
print(f"Saved: {out}")

# ---- Fig 15 style: off-diagonal distributions, pooled across trials ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
bins_log = np.logspace(-5, 0, 100)
for d, label, color in [(pooled_offdiag_base_vals, "Base", "gray"), (pooled_offdiag_cell_vals, "Cellular", "green")]:
    sns.histplot(d, bins=bins_log, stat="probability", element="step", label=label, color=color, ax=axes[0])
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_title("Probability of Individual Weights"); axes[0].set_xlabel("Attention Weight"); axes[0].set_ylabel("Probability")
axes[0].legend()
bins_lin = np.linspace(0, 1, 50)
for d, label, color in [(pooled_offdiag_base_mass, "Base", "gray"), (pooled_offdiag_cell_mass, "Cellular", "green")]:
    sns.histplot(d, bins=bins_lin, stat="probability", element="step", label=label, color=color, ax=axes[1])
axes[1].set_title("Context Attention Mass per Token"); axes[1].set_xlabel("Off-Diagonal Mass (Sum per row)"); axes[1].set_ylabel("Probability")
axes[1].legend()
sns.ecdfplot(pooled_offdiag_base_mass, label="Base", color="gray", ax=axes[2])
sns.ecdfplot(pooled_offdiag_cell_mass, label="Cellular", color="green", ax=axes[2])
axes[2].set_title("Cumulative Distribution (CDF)"); axes[2].set_xlabel("Off-Diagonal Mass"); axes[2].set_ylabel("Cumulative Probability")
axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "off_diagonal_probability_multitrial.png")
plt.savefig(out, dpi=300); plt.close(fig)
print(f"Saved: {out}")

# ---- Fig 16 style: distance decay, averaged across trials ----
palette = {
    "Base (No Dividers)": "#7f8c8d", "Cellular (Global)": "#2ecc71",
    "Cellular (To Divider)": "#e74c3c", "Cellular (Alpha-to-Alpha)": "#3498db"
}
plt.figure(figsize=(12, 7))
sns.lineplot(data=distdecay_df, x="Distance", y="Attention", hue="Type", palette=palette, linewidth=2.5, errorbar='se')
max_dist = int(distdecay_df["Distance"].max())
for d in range(3, max_dist + 1, 3):
    plt.axvline(x=d, color='red', linestyle='--', alpha=0.3)
plt.title(f"Attention Profile, averaged over {N_TRIALS} trials (shaded = standard error)", fontsize=14)
plt.yscale('log')
plt.xticks(range(1, max_dist + 1))
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "attention_distance_profile_multitrial.png")
plt.savefig(out, dpi=300); plt.close()
print(f"Saved: {out}")

# Save the raw numbers too
summary = {
    "n_trials": N_TRIALS,
    "mean_entropy_base": float(mean_entropy_base.mean()),
    "mean_entropy_cell": float(mean_entropy_cell.mean()),
    "cell_higher_layers": int((mean_entropy_cell.mean(axis=1) > mean_entropy_base.mean(axis=1)).sum()),
    "n_layers": num_layers,
    "mean_offdiag_mass_base": float(pooled_offdiag_base_mass.mean()),
    "mean_offdiag_mass_cell": float(pooled_offdiag_cell_mass.mean()),
    "boundary_peak_layer": peak_layer,
    "boundary_peak_value": float(per_layer_mean[peak_layer]),
}
pd.Series(summary).to_csv(os.path.join(RESULTS_DIR, "multitrial_attention_summary.csv"))
print("\nSaved summary stats. Multi-trial figure regeneration complete.")
