#!/usr/bin/env python3
"""Figure 2 (main paper, ICLR restructure): the full attention-sink mechanism
story in one four-panel figure. Panels (a)-(c) are the former Figure 1
(boundary-token attention mass vs. density, a representative attending head,
and the causal ablation-by-length result). Panel (d) is the former Figure 3
panel (a): attention to the actual needed token collapses specifically at
the misaligned boundary density (copy_previous, N=32) -- the mechanistic
half of the dip story, now grouped with the rest of the attention-sink
evidence rather than split across two figures.

Qwen2.5-32B-Instruct throughout. Panels (a)-(c): cyclic_shift, real
(concatenated) letter tokenization. Panel (d): copy_previous.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, set_power_of_2_xticks

set_rc()

# Validated palette (dataviz skill, references/palette.md)
SEQ_BLUE_ORDINAL = ["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
CAT_BLUE = "#2a78d6"
CAT_ORANGE = "#eb6834"
CAT_AQUA = "#1baf7a"

RESULTS_DIR = REPO_ROOT / "results" / "Qwen2.5-32B-Instruct"

fig = plt.figure(figsize=(MULTI_COL_WIDTH * 2.0, 3.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.5, 1.0, 0.9])

# ---- Panel (a): boundary attention mass vs density, faceted by length ----
sweep = pd.read_csv(RESULTS_DIR / "attention_metrics_32b_realletters.csv")
npz = np.load(RESULTS_DIR / "attention_run_0_32b_realletters.npz", allow_pickle=True)
attn_cell = npz["attn_cell"]
tokens_cell = [str(t) for t in npz["tokens_cell"]]

ax_a = fig.add_subplot(gs[0])
sweep_a2 = sweep[sweep.Alphabet_Size == 2] if "Alphabet_Size" in sweep.columns else sweep
sweep_a2 = sweep_a2[sweep_a2.Interval != 3]
lengths = sorted(sweep_a2.Seq_Length.unique())
for color, N in zip(SEQ_BLUE_ORDINAL, lengths):
    sub = sweep_a2[sweep_a2.Seq_Length == N]
    g = sub.groupby("Inv_K")["Cell_Boundary_Attn"].agg(["mean", "sem"])
    ax_a.errorbar(g.index, g["mean"], yerr=g["sem"], color=color, marker="o",
                  markersize=4, linewidth=1.5, capsize=2, label=f"N={N}")
ax_a.set_xlabel("Boundary density (1/k)")
ax_a.set_ylabel("Attention mass on boundary tokens")
ax_a.set_title("(a) Boundary tokens absorb attention,\nscaling with density (alphabet=2)", fontsize=8.5)
set_power_of_2_xticks(ax_a, sweep_a2.Inv_K.unique(), fontsize=6.5)
ax_a.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6.5, ncol=1)

# ---- Panel (b): representative heatmap, most boundary-attending head ----
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

# ---- Panel (c): causal ablation-by-length ----
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
set_power_of_2_xticks(ax_c, ablation_N)
ax_c.set_ylim(0.4, 1.05)
ax_c.set_xlabel("Sequence length N")
ax_c.set_ylabel("TAR")
ax_c.set_title("(c) Blocking sink heads erases a growing\nshare of the benefit as N increases", fontsize=8.5)
ax_c.legend(loc="lower left", frameon=False, fontsize=6)

# ---- Panel (d): dip mechanism -- attention to correct target (from former Fig 3a) ----
ax_d = fig.add_subplot(gs[3])
retrieval = pd.read_csv(RESULTS_DIR / "dip_explanation_retrieval_attn.csv")
top10 = retrieval.groupby("k")["correct_target_attn"].apply(lambda s: s.sort_values(ascending=False).head(10).mean())
k_labels = {1: "k=1\n(regular)", 2: "k=2\n(the dip)", 8: "k=8\n(recovered)"}
ks = [1, 2, 8]
colors = [CAT_BLUE, CAT_ORANGE, CAT_AQUA]
bars = ax_d.bar([k_labels[k] for k in ks], [top10[k] for k in ks], color=colors, width=0.6)
for bar, k in zip(bars, ks):
    ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{top10[k]:.3f}",
               ha="center", fontsize=8)
ax_d.set_ylabel("Mean attention to correct target\n(top-10 heads)")
ax_d.set_title("(d) Attention to the needed token collapses\nat the misaligned density\n(copy_previous, N=32)", fontsize=8.5)
ax_d.set_ylim(0, 0.65)

fig.suptitle("Boundary markers act as attention sinks: mechanism and evidence (Qwen2.5-32B-Instruct)",
             fontsize=9.5, y=1.06)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "fig2_mechanism_merged.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Best boundary-attending head: L{l}/H{h}, score={best_score:.3f}")
