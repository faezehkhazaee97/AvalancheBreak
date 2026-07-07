import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys

MODEL_TAG = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-7B-Instruct"
RESULTS_DIR = os.path.join("results", MODEL_TAG)

npz_path = os.path.join(RESULTS_DIR, "attention_run_0_templated.npz")
data = np.load(npz_path, allow_pickle=True)

attn_base = data["attn_base"]
attn_cell = data["attn_cell"]
tokens_base = np.array(data["tokens_base"])
tokens_cell = np.array(data["tokens_cell"])

num_layers, num_heads = attn_base.shape[:2]
print(f"[{MODEL_TAG}] Loaded {npz_path}: {num_layers} layers x {num_heads} heads")
print(f"tokens_base: {list(tokens_base)}")
print(f"tokens_cell: {list(tokens_cell)}")

mask_cell = np.array([t != '|' for t in tokens_cell])
attn_cell_masked = attn_cell[:, :, mask_cell, :][:, :, :, mask_cell].copy()
tokens_cell_masked = np.array(tokens_cell)[mask_cell]

sns.set_theme(style="whitegrid", font_scale=1.1)

# ============================================================
# FIGURE 2: Example attention matrix (Layer 13, Head 10 -- matches paper)
# ============================================================
layer_idx, head_idx = 13, 10
if layer_idx < num_layers and head_idx < num_heads:
    m_base = attn_base[layer_idx, head_idx]
    m_cell = attn_cell_masked[layer_idx, head_idx]
    vmax_shared = max(np.max(m_base), np.max(m_cell))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(m_base, xticklabels=tokens_base, yticklabels=tokens_base, cmap="rocket",
                vmin=0, vmax=vmax_shared, ax=ax1)
    ax1.set_title(f"Base Attention\nLayer {layer_idx}, Head {head_idx}", fontsize=12)
    ax1.set_aspect('equal')
    sns.heatmap(m_cell, xticklabels=tokens_cell_masked, yticklabels=tokens_cell_masked, cmap="rocket",
                vmin=0, vmax=vmax_shared, ax=ax2)
    ax2.set_title(f"Cellular Attention (Alphabetic Only)\nLayer {layer_idx}, Head {head_idx}", fontsize=12)
    ax2.set_aspect('equal')
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "example_attention_matrix_templated.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 2: {out}")
else:
    print(f"Skipping Figure 2: layer/head {layer_idx}/{head_idx} out of range for {num_layers}x{num_heads}")


# ============================================================
# FIGURE 3: Boundary attention sink (per-layer line + heads heatmap)
# ============================================================
boundary_idx = np.where(tokens_cell == '|')[0]
if len(boundary_idx) > 0:
    per_layer_head_boundary = np.zeros((num_layers, num_heads))
    for l in range(num_layers):
        for h in range(num_heads):
            m = attn_cell[l, h]  # unmasked -- need boundary columns present
            per_layer_head_boundary[l, h] = np.mean(np.sum(m[:, boundary_idx], axis=-1))

    per_layer_mean = per_layer_head_boundary.mean(axis=1)
    peak_layer = int(np.argmax(per_layer_mean))
    peak_val = per_layer_mean[peak_layer]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.plot(range(num_layers), per_layer_mean, marker='o', markersize=5, color='#2b6cb0')
    ax.annotate(f'Spike at Layer {peak_layer}', xy=(peak_layer, peak_val),
                xytext=(peak_layer, peak_val + 0.02 * max(per_layer_mean)),
                ha='center', color='crimson', fontsize=10, fontweight='bold')
    ax.scatter([peak_layer], [peak_val], color='crimson', zorder=5, s=60)
    ax.set_xlabel("Model Layer Depth")
    ax.set_ylabel("Mean Boundary Attention Mass")
    ax.set_title("Average Attention Absorbed by Boundaries per Layer", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sns.heatmap(per_layer_head_boundary.T, cmap="rocket", ax=ax,
                cbar_kws={'label': 'Attention Mass -> Boundaries'})
    ax.set_xlabel("Model Layer Depth")
    ax.set_ylabel("Attention Head Index")
    ax.set_title("Boundary Attention Heatmap (All Heads)", fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "boundary_sink_analysis_templated.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 3: {out} (peak at layer {peak_layer}, {peak_val*100:.1f}% of attention mass)")
else:
    print("Skipping Figure 3: no boundary tokens found in tokens_cell")


# ============================================================
# FIGURE 4: Head entropy (layer-wise mean + distribution)
# ============================================================
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

layer_mean_base = np.mean(entropy_base, axis=1)
layer_mean_cell = np.mean(entropy_cell, axis=1)
layer_mean_delta = np.mean(delta_entropy, axis=1)

fig = plt.figure(figsize=(8, 5))
plt.plot(layer_mean_base, label="Base")
plt.plot(layer_mean_cell, label="Cellular")
plt.plot(layer_mean_delta, label="Delta (Cell-Base)", linestyle="--")
plt.xlabel("Layer")
plt.ylabel("Mean Head Entropy")
plt.title("Layer-wise Mean Attention Entropy (Alphabetic Only)")
plt.legend()
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "head_entropy_layerwise_templated.png")
plt.savefig(out, dpi=300)
plt.close(fig)
print(f"Saved Figure 4a: {out}")

fig = plt.figure(figsize=(8, 5))
plt.hist(entropy_base.flatten(), bins=40, alpha=0.6, label="Base")
plt.hist(entropy_cell.flatten(), bins=40, alpha=0.6, label="Cellular")
plt.xlabel("Head Entropy")
plt.ylabel("Frequency")
plt.title("Distribution of Head Entropy (Alphabetic Only)")
plt.legend()
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "head_entropy_distribution_templated.png")
plt.savefig(out, dpi=300)
plt.close(fig)
print(f"Saved Figure 4b: {out}")


# ============================================================
# FIGURE 5: Off-diagonal attention distributions
# ============================================================
def offdiag_fraction(attn_tensor):
    L, H = attn_tensor.shape[:2]
    frac = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            m = attn_tensor[l, h]
            off = np.tril(m, k=-1).sum(axis=1)
            frac[l, h] = off[1:].mean()
    return frac

def offdiag_values(attn_tensor):
    L, H = attn_tensor.shape[:2]
    vals = []
    for l in range(L):
        for h in range(H):
            m = attn_tensor[l, h]
            N = m.shape[0]
            for i in range(N):
                off = np.delete(m[i, :], i)
                vals.extend(off)
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

ratio_base = offdiag_fraction(attn_base)
ratio_cell = offdiag_fraction(attn_cell_masked)
print(f"\nMean Base context mass: {np.mean(ratio_base):.4f}")
print(f"Mean Cellular context mass: {np.mean(ratio_cell):.4f}")

off_base_vals = offdiag_values(attn_base)
off_cell_vals = offdiag_values(attn_cell_masked)
off_base_mass = offdiag_mass_per_row(attn_base)
off_cell_mass = offdiag_mass_per_row(attn_cell_masked)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
bins_log = np.logspace(-5, 0, 100)
for d, label, color in [(off_base_vals, "Base", "gray"), (off_cell_vals, "Cellular", "green")]:
    sns.histplot(d, bins=bins_log, stat="probability", element="step", label=label, color=color, ax=axes[0])
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_title("Probability of Individual Weights")
axes[0].set_xlabel("Attention Weight"); axes[0].set_ylabel("Probability")
axes[0].legend()

bins_lin = np.linspace(0, 1, 50)
for d, label, color in [(off_base_mass, "Base", "gray"), (off_cell_mass, "Cellular", "green")]:
    sns.histplot(d, bins=bins_lin, stat="probability", element="step", label=label, color=color, ax=axes[1])
axes[1].set_title("Context Attention Mass per Token")
axes[1].set_xlabel("Off-Diagonal Mass (Sum per row)"); axes[1].set_ylabel("Probability")
axes[1].legend()

sns.ecdfplot(off_base_mass, label="Base", color="gray", ax=axes[2])
sns.ecdfplot(off_cell_mass, label="Cellular", color="green", ax=axes[2])
axes[2].set_title("Cumulative Distribution (CDF)")
axes[2].set_xlabel("Off-Diagonal Mass"); axes[2].set_ylabel("Cumulative Probability")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "off_diagonal_probability_templated.png")
plt.savefig(out, dpi=300)
plt.close(fig)
print(f"Saved Figure 5: {out}")


# ============================================================
# FIGURE 6: Distance-decay attention profile
# ============================================================
attn_base_avg = attn_base.mean(axis=(0, 1))
attn_cell_avg = attn_cell.mean(axis=(0, 1))

palette = {
    "Base (No Dividers)": "#7f8c8d",
    "Cellular (Global)": "#2ecc71",
    "Cellular (To Divider)": "#e74c3c",
    "Cellular (Alpha-to-Alpha)": "#3498db"
}

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

df = pd.DataFrame(results)
plt.figure(figsize=(12, 7))
sns.lineplot(data=df, x="Distance", y="Attention", hue="Type", palette=palette, linewidth=2.5)
max_dist = int(df["Distance"].max())
for d in range(3, max_dist + 1, 3):
    plt.axvline(x=d, color='red', linestyle='--', alpha=0.3)
plt.title("Attention Profile: Notice the 'Dips' at Divider Intervals (Templated)", fontsize=14)
plt.yscale('log')
plt.xticks(range(1, max_dist + 1))
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "attention_distance_profile_templated.png")
plt.savefig(out, dpi=300)
plt.close()
print(f"Saved Figure 6: {out}")

print("\nAll attention figures regenerated from templated data.")
