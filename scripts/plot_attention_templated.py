import matplotlib
matplotlib.use("Agg")

# ============================================================
# PLOT FIGURES 2-6 FROM THE TEMPLATED ATTENTION CAPTURE
# ============================================================
# Reads attention_run_0_templated.npz and attention_metrics_templated.csv
# (produced by run_attention_templated.py) and reproduces the same five
# figures as the raw-prompt run, using identical metric computations
# (entropy, off-diagonal mass, distance-decay, boundary-attention
# regression) so the two are a fair side-by-side comparison. CPU-only,
# no model load needed.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)

GREY = "#7f8c8d"
GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#3498db"

# ============================================================
# FIGURE 2: example attention matrix, Layer 13 Head 10
# ============================================================
data = np.load(os.path.join(RESULTS_DIR, "attention_run_0_templated.npz"), allow_pickle=True)
attn_base = data["attn_base"]
attn_cell = data["attn_cell"]
tokens_base = np.array(data["tokens_base"])
tokens_cell = np.array(data["tokens_cell"])

LAYER, HEAD = 13, 10
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, mat, toks, title in [
    (axes[0], attn_base[LAYER, HEAD], tokens_base, f"Base Attention (templated)\nLayer {LAYER}, Head {HEAD}"),
    (axes[1], attn_cell[LAYER, HEAD], tokens_cell, f"Cellular Attention (templated)\nLayer {LAYER}, Head {HEAD}"),
]:
    im = ax.imshow(mat, cmap="magma", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks)
    ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "example_attention_matrix_templated.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved example_attention_matrix_templated.png")

# ============================================================
# FIGURE 4: entropy (layer-wise mean + distribution)
# ============================================================
mask_cell = np.array([t != '|' for t in tokens_cell])
attn_cell_masked = attn_cell[:, :, mask_cell, :][:, :, :, mask_cell]

def compute_head_entropy(attn_tensor):
    eps = 1e-12
    L, H = attn_tensor.shape[:2]
    entropy_matrix = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            m = attn_tensor[l, h]
            row_sums = m.sum(axis=-1, keepdims=True)
            m_norm = m / (row_sums + eps)
            row_entropy = -np.sum(m_norm * np.log2(m_norm + eps), axis=-1)
            entropy_matrix[l, h] = np.mean(row_entropy)
    return entropy_matrix

entropy_base = compute_head_entropy(attn_base)
entropy_cell = compute_head_entropy(attn_cell_masked)
delta_entropy = entropy_cell - entropy_base

layer_mean_base = entropy_base.mean(axis=1)
layer_mean_cell = entropy_cell.mean(axis=1)
layer_mean_delta = delta_entropy.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(layer_mean_base, label="Base", color=BLUE)
plt.plot(layer_mean_cell, label="Cellular", color="#e67e22")
plt.plot(layer_mean_delta, label="Delta (Cell-Base)", linestyle="--", color=GREEN)
plt.xlabel("Layer"); plt.ylabel("Mean Head Entropy")
plt.title("Layer-wise Mean Attention Entropy (templated)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "head_entropy_layerwise_templated.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved head_entropy_layerwise_templated.png")

plt.figure(figsize=(8, 5))
plt.hist(entropy_base.flatten(), bins=40, alpha=0.6, label="Base", color=BLUE)
plt.hist(entropy_cell.flatten(), bins=40, alpha=0.6, label="Cellular", color="#e67e22")
plt.xlabel("Head Entropy"); plt.ylabel("Frequency")
plt.title("Distribution of Head Entropy (templated)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "head_entropy_distribution_templated.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved head_entropy_distribution_templated.png")

print(f"Mean entropy delta (Cell-Base), templated: {delta_entropy.mean():+.4f} "
      f"(raw-prompt run's own delta was negative at nearly every layer -- compare directly)")

# ============================================================
# FIGURE 5: off-diagonal probability (3-panel)
# ============================================================
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

off_base_vals = offdiag_values(attn_base)
off_cell_vals = offdiag_values(attn_cell_masked)
off_base_mass = offdiag_mass_per_row(attn_base)
off_cell_mass = offdiag_mass_per_row(attn_cell_masked)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
bins_log = np.logspace(-5, 0, 100)
for vals, label, color in [(off_base_vals, "Base", GREY), (off_cell_vals, "Cellular", GREEN)]:
    axes[0].hist(vals, bins=bins_log, weights=np.ones_like(vals) / len(vals),
                 histtype="step", label=label, color=color, linewidth=1.8)
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_title("Probability of Individual Weights (templated)")
axes[0].set_xlabel("Attention Weight"); axes[0].set_ylabel("Probability"); axes[0].legend()

bins_lin = np.linspace(0, 1, 50)
for vals, label, color in [(off_base_mass, "Base", GREY), (off_cell_mass, "Cellular", GREEN)]:
    axes[1].hist(vals, bins=bins_lin, weights=np.ones_like(vals) / len(vals),
                 histtype="step", label=label, color=color, linewidth=1.8)
axes[1].set_title("Context Attention Mass per Token (templated)")
axes[1].set_xlabel("Off-Diagonal Mass (Sum per row)"); axes[1].set_ylabel("Probability"); axes[1].legend()

for vals, label, color in [(off_base_mass, "Base", GREY), (off_cell_mass, "Cellular", GREEN)]:
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axes[2].plot(sorted_vals, cdf, label=label, color=color, linewidth=1.8)
axes[2].set_title("Cumulative Distribution (templated)")
axes[2].set_xlabel("Off-Diagonal Mass"); axes[2].set_ylabel("Cumulative Probability")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "off_diagonal_probability_templated.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved off_diagonal_probability_templated.png")

# ============================================================
# FIGURE 6: distance-decay (Base / Cellular-Global / To-Divider / Alpha-to-Alpha)
# ============================================================
attn_base_avg = data["attn_base"].mean(axis=(0, 1))
attn_cell_avg = data["attn_cell"].mean(axis=(0, 1))

results = []
N_base = attn_base_avg.shape[0]
for i in range(1, N_base):
    for j in range(i):
        results.append({"Distance": i - j, "Attention": attn_base_avg[i, j], "Type": "Base (No Dividers)"})

N_cell = attn_cell_avg.shape[0]
div_idx = np.where(tokens_cell == "|")[0]
alpha_idx = np.where(tokens_cell != "|")[0]
for i in range(1, N_cell):
    for j in range(i):
        dist = i - j
        val = attn_cell_avg[i, j]
        results.append({"Distance": dist, "Attention": val, "Type": "Cellular (Global)"})
        if j in div_idx:
            results.append({"Distance": dist, "Attention": val, "Type": "Cellular (To Divider)"})
        if i in alpha_idx and j in alpha_idx:
            results.append({"Distance": dist, "Attention": val, "Type": "Cellular (Alpha-to-Alpha)"})

df_dist = pd.DataFrame(results)
summary = df_dist.groupby(["Distance", "Type"])["Attention"].agg(["mean", "std"]).reset_index()

palette = {
    "Base (No Dividers)": GREY,
    "Cellular (Global)": GREEN,
    "Cellular (To Divider)": RED,
    "Cellular (Alpha-to-Alpha)": BLUE,
}
plt.figure(figsize=(12, 7))
for t, color in palette.items():
    sub = summary[summary["Type"] == t]
    plt.plot(sub["Distance"], sub["mean"], label=t, color=color, linewidth=2.2)
    plt.fill_between(sub["Distance"], sub["mean"] - sub["std"].fillna(0), sub["mean"] + sub["std"].fillna(0),
                      color=color, alpha=0.15)
for d in range(k_interval := data["k_interval"].item(), 21, k_interval):
    plt.axvline(x=d, color="red", linestyle="--", alpha=0.3)
plt.yscale("log")
plt.xticks(range(1, 21))
plt.title("Attention Profile vs Token Distance (templated)")
plt.xlabel("Distance"); plt.ylabel("Attention")
plt.legend(title="Type")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "attention_distance_profile_templated.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved attention_distance_profile_templated.png")

# ============================================================
# FIGURE 3: boundary attention vs 1/k, faceted by sequence length
# ============================================================
df = pd.read_csv(os.path.join(RESULTS_DIR, "attention_metrics_templated.csv"))
seq_lengths = sorted(df["Seq_Length"].unique())
fig, axes = plt.subplots(1, len(seq_lengths), figsize=(4 * len(seq_lengths), 5), sharex=True, sharey=True)
for i, N in enumerate(seq_lengths):
    ax = axes[i]
    sub_N = df[df["Seq_Length"] == N]
    summary_N = sub_N.groupby("Inv_K")["Cell_Boundary_Attn"].agg(["mean", "std"]).reset_index()
    ax.scatter(sub_N["Inv_K"], sub_N["Cell_Boundary_Attn"], color=RED, alpha=0.3, s=25, label="Trials")
    ax.errorbar(summary_N["Inv_K"], summary_N["mean"], yerr=summary_N["std"].fillna(0),
                fmt="o", color=RED, markersize=7, capsize=4, label="Mean +/- std")
    if sub_N["Inv_K"].nunique() >= 2:
        slope, intercept, r, p, se = linregress(sub_N["Inv_K"], sub_N["Cell_Boundary_Attn"])
        xf = np.linspace(0, sub_N["Inv_K"].max() * 1.1, 50)
        ax.plot(xf, slope * xf + intercept, linestyle="--", color=RED, alpha=0.6)
    ax.set_title(f"Length N = {N}", fontweight="bold")
    ax.set_xlabel("1/k (boundary frequency)")
    if i == 0:
        ax.set_ylabel("Mean Attention on Boundary Tokens")
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "boundary_sink_analysis_templated.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved boundary_sink_analysis_templated.png")

print("\nAll templated mechanistic figures generated.")
