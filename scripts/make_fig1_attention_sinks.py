#!/usr/bin/env python3
"""Figure 1 (main paper): boundary markers act as attention sinks, and
that sink behavior is causally necessary for the benefit. Qwen2.5-32B-
Instruct, cyclic_shift, REAL (concatenated) letter tokenization -- matches
the actual accuracy-grid prompt format exactly, no space-separation
workaround. Three panels: (a) boundary-token attention mass vs. density,
faceted by length, alphabet=2 (spot-checked against alphabet=8 in the
sweep data -- see printed regression slopes) -- shows the sink effect
scales with how many markers are inserted; (b) one representative
attention heatmap (persistent boundary-attending head) showing what the
sink looks like; (c) causal confirmation -- blocking the top boundary-
attending heads' access to the boundary tokens erases a growing fraction
of the accuracy benefit as N increases, now including N=128.

Palette: validated dataviz-skill palette (references/palette.md). Panel
(a) uses the sequential blue ramp (ordinal steps, since N is a discrete
ordered variable, not a continuum) instead of viridis, for consistency
with the rest of the palette; panel (c) uses categorical slots 1 (blue,
intact/normal) and 2 (orange, disrupted/ablated), keeping Base as
black-dashed per this report's own established convention everywhere else.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH

set_rc()

# Validated palette (dataviz skill, references/palette.md)
SEQ_BLUE_ORDINAL = ["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]  # steps 250..700
CAT_BLUE = "#2a78d6"    # slot 1
CAT_ORANGE = "#eb6834"  # slot 2

RESULTS_DIR = REPO_ROOT / "results" / "Qwen2.5-32B-Instruct"
sweep = pd.read_csv(RESULTS_DIR / "attention_metrics_32b_realletters.csv")
npz = np.load(RESULTS_DIR / "attention_run_0_32b_realletters.npz", allow_pickle=True)
attn_cell = npz["attn_cell"]
tokens_cell = [str(t) for t in npz["tokens_cell"]]

fig = plt.figure(figsize=(MULTI_COL_WIDTH * 1.5, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.0])

# Panel (a): boundary attention mass vs density, faceted by length, alphabet=2
ax_a = fig.add_subplot(gs[0])
sweep_a2 = sweep[sweep.Alphabet_Size == 2] if "Alphabet_Size" in sweep.columns else sweep
lengths = sorted(sweep_a2.Seq_Length.unique())
for color, N in zip(SEQ_BLUE_ORDINAL, lengths):
    sub = sweep_a2[sweep_a2.Seq_Length == N]
    g = sub.groupby("Inv_K")["Cell_Boundary_Attn"].agg(["mean", "sem"])
    ax_a.errorbar(g.index, g["mean"], yerr=g["sem"], color=color, marker="o",
                  markersize=4, linewidth=1.5, capsize=2, label=f"N={N}")
ax_a.set_xlabel("Boundary density (1/k)")
ax_a.set_ylabel("Attention mass on boundary tokens")
ax_a.set_title("(a) Boundary tokens absorb attention,\nscaling with density (alphabet=2)", fontsize=8.5)
ax_a.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6.5, ncol=1)

# Panel (b): representative heatmap, most boundary-attending head
n_layers, n_heads = attn_cell.shape[0], attn_cell.shape[1]
boundary_idx = [i for i, t in enumerate(tokens_cell) if "|" in t]
best_score, best_lh = -1, (0, 0)
for l in range(n_layers):
    for h in range(n_heads):
        m = attn_cell[l, h]
        score = m[:, boundary_idx].sum() if boundary_idx else 0
        if score > best_score:
            best_score, best_lh = score, (l, h)
l, h = best_lh
ax_b = fig.add_subplot(gs[1])
m = attn_cell[l, h]
im = ax_b.imshow(m, cmap="viridis", vmin=0, vmax=np.percentile(m, 99), aspect="auto")
ax_b.set_xticks(range(len(tokens_cell)))
ax_b.set_xticklabels(tokens_cell, rotation=90, fontsize=8)
ax_b.set_yticks(range(len(tokens_cell)))
ax_b.set_yticklabels(tokens_cell, fontsize=8)
ax_b.tick_params(axis="x", pad=2)
ax_b.set_xlabel("Key token (attended TO)", fontsize=7)
ax_b.set_ylabel("Query token (attending FROM)", fontsize=7)
for i in boundary_idx:
    ax_b.axvline(i, color=CAT_ORANGE, alpha=0.5, linewidth=0.8)
    ax_b.axhline(i, color=CAT_ORANGE, alpha=0.5, linewidth=0.8)
ax_b.set_title(f"(b) Attention heatmap, L{l}/H{h}\n(orange lines = boundary tokens)", fontsize=8.5)
fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)

# Panel (c): causal ablation-by-length (now includes N=128)
ax_c = fig.add_subplot(gs[2])
abl_csv = RESULTS_DIR / "causal_ablation_grid.csv"
base_grid = pd.read_csv(RESULTS_DIR / "thermodynamics_grid_templated_v2.csv")
abl = pd.read_csv(abl_csv)
ablation_N = sorted(abl.Seq_Length.unique())
ref = base_grid[(base_grid.Alphabet_Size == 2) & (base_grid.Seq_Length.isin(ablation_N))]
base_tar = ref[ref.Method == "Base"].groupby("Seq_Length")["TAR"].mean().reindex(ablation_N)
normal_tar = ref[(ref.Method == "Cellular_Bounds") & (ref.Interval == 2)].groupby("Seq_Length")["TAR"].mean().reindex(ablation_N)
ablated_tar = abl.groupby("Seq_Length")["TAR"].mean().reindex(ablation_N)

ax_c.plot(ablation_N, base_tar, color="black", linestyle="--", marker="o", markersize=4, linewidth=1.3, label="Base")
ax_c.plot(ablation_N, normal_tar, color=CAT_BLUE, marker="s", markersize=4, linewidth=1.6, label="Cellular_Bounds")
ax_c.plot(ablation_N, ablated_tar, color=CAT_ORANGE, marker="^", markersize=4, linewidth=1.6, label="Cellular_Bounds\n(sink heads ablated)")
ax_c.set_xscale("log", base=2)
ax_c.set_xticks(ablation_N)
ax_c.set_xticklabels(ablation_N)
ax_c.set_ylim(0.4, 1.05)
ax_c.set_xlabel("Sequence length N")
ax_c.set_ylabel("TAR")
ax_c.set_title("(c) Blocking sink heads erases a growing\nshare of the benefit as N increases", fontsize=8.5)
ax_c.legend(loc="lower left", frameon=False, fontsize=6)

fig.suptitle("Boundary markers act as attention sinks, and that mechanism is causally necessary\n(Qwen2.5-32B, cyclic_shift, alphabet=2, real tokenization)",
             fontsize=9, y=1.06)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "fig1_attention_sinks.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Best boundary-attending head: L{l}/H{h}, score={best_score:.3f}")
print("\nPanel (c) values:")
print(pd.DataFrame({"Base": base_tar, "Normal": normal_tar, "Ablated": ablated_tar,
                     "Pct_erased": (normal_tar - ablated_tar) / (normal_tar - base_tar)}).round(4))

if "Alphabet_Size" in sweep.columns and 8 in sweep.Alphabet_Size.unique():
    print("\nAlphabet=8 spot-check (not plotted, sanity check only):")
    sweep_a8 = sweep[sweep.Alphabet_Size == 8]
    from scipy.stats import linregress
    for N, sub in sweep_a8.groupby("Seq_Length"):
        if sub["Inv_K"].nunique() >= 2:
            slope, intercept, r, p, se = linregress(sub["Inv_K"], sub["Cell_Boundary_Attn"])
            print(f"  N={N:>4}: slope={slope:.4f}  R^2={r**2:.4f}")
