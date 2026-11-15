#!/usr/bin/env python3
"""Figure 2 (main paper, ICLR restructure): the full attention-sink mechanism
story in one four-panel figure. Panels (a)-(c) are the former Figure 1
(boundary-token attention mass vs. density, a representative attending head,
and the causal ablation-by-length result). Panel (d) is the former Figure 3
panel (a): attention to the actual needed token collapses specifically at
the misaligned boundary density (copy_previous, N=32) -- the mechanistic
half of the dip story, now grouped with the rest of the attention-sink
evidence rather than split across two figures.

Qwen2.5-32B-Instruct throughout. Panels (a)-(b): cyclic_shift, real
(concatenated) letter tokenization. Panel (c) (replacing the earlier, uncontrolled
cyclic_shift ablation-by-length): the controlled head-count dose response on
random_permutation with spaced letters and digits (Table tab:dose). Panel (d): copy_previous.
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
SEQ_BLUE_ORDINAL = ["#b9d7ee", "#9cc4e6", "#6faad8", "#0072B2", "#2b6ea8", "#0072B2", "#003f6b"]
CAT_BLUE = "#0072B2"
CAT_ORANGE = "#D55E00"
CAT_AQUA = "#009E73"

RESULTS_DIR = REPO_ROOT / "results" / "Qwen2.5-32B-Instruct"

fig = plt.figure(figsize=(MULTI_COL_WIDTH, 5.8))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], hspace=0.45, wspace=0.42)

# ---- Panel (a): boundary attention mass vs density, faceted by length ----
sweep = pd.read_csv(RESULTS_DIR / "attention_metrics_32b_realletters.csv")
npz = np.load(RESULTS_DIR / "attention_run_0_32b_realletters.npz", allow_pickle=True)
attn_cell = npz["attn_cell"]
tokens_cell = [str(t) for t in npz["tokens_cell"]]

ax_a = fig.add_subplot(gs[0, 0])
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
ax_a.set_title("(a) Attention on all markers, by density", fontsize=8.5)
set_power_of_2_xticks(ax_a, sweep_a2.Inv_K.unique(), fontsize=6.5)
ax_a.legend(loc="upper left", frameon=False, fontsize=6, ncol=1, borderaxespad=0.2, labelspacing=0.25)

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
ax_b = fig.add_subplot(gs[0, 1])
m = attn_cell[l, h]
im = ax_b.imshow(m, cmap="viridis", vmin=0, vmax=np.percentile(m, 99), aspect="auto")
# Horizontal tick labels (tokens are at most four characters), the marker shown as a real bar
# in orange like its grid lines, so each label stands apart from the letter tokens beside it.
ax_b.set_xticks(range(len(tokens_cell)))
ax_b.set_xticklabels(tokens_cell, rotation=0, fontsize=6.5)
ax_b.set_yticks(range(len(tokens_cell)))
ax_b.set_yticklabels(tokens_cell, fontsize=6.5)
for axis in (ax_b.xaxis, ax_b.yaxis):
    for tick, tok in zip(axis.get_major_ticks(len(tokens_cell)), tokens_cell):
        if "|" in tok:
            tick.label1.set_color(CAT_ORANGE)
ax_b.tick_params(axis="x", pad=2)
ax_b.grid(False)
ax_b.set_xlabel("Key token (attended to)", fontsize=7)
ax_b.set_ylabel("Query token (attending from)", fontsize=7)
for i in boundary_idx:
    ax_b.axvline(i, color=CAT_ORANGE, alpha=0.5, linewidth=0.8)
    ax_b.axhline(i, color=CAT_ORANGE, alpha=0.5, linewidth=0.8)
ax_b.set_title(f"(b) Head L{l}/H{h} (orange lines: markers)", fontsize=8.5)
fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)

# ---- Panel (c): controlled head-blocking dose response (Table tab:dose) ----
# results_clean/_summary/head_dose.csv (scripts/analyze_wording_and_dose.py): Qwen2.5-32B,
# random_permutation, A = 8, N = 64, k = 8, 20 trials; share of the marker gain (Bounds - Base)
# removed by blocking the top-n marker-attending heads from the markers, n random heads from the
# markers, or the top-n heads from as many random non-marker tokens (sham). Bars: 95% intervals.
dose = pd.read_csv(REPO_ROOT / "results_clean" / "_summary" / "head_dose.csv")
dose["gain"] = dose.bounds - dose.base
dose["share_ci"] = dose.ci / dose.gain
gs_c = gs[1, 0].subgridspec(1, 2, wspace=0.12)
ax_c = fig.add_subplot(gs_c[0, 0])
ax_c2 = fig.add_subplot(gs_c[0, 1], sharey=ax_c)
DOSE_STYLE = {"ablate_top": ("Top heads", CAT_ORANGE, "o", "-"),
              "ctrl_random": ("Random heads", "#6e6e6e", "s", "--"),
              "ctrl_sham": ("Sham (top heads,\nnon-marker tokens)", CAT_AQUA, "^", ":")}
DODGE = {"ablate_top": 0.94, "ctrl_random": 1.0, "ctrl_sham": 1.06}
for axx, (task, fmt_label) in zip([ax_c, ax_c2], [("random_permutation_spaced", "Spaced letters"),
                                                   ("random_permutation_digits", "Digits")]):
    axx.axhline(0, color="#6e6e6e", linewidth=0.6, zorder=1)
    for cond, (lab, color, mk, ls) in DOSE_STYLE.items():
        d = dose[(dose.task == task) & (dose.condition == cond)].sort_values("heads")
        axx.errorbar(d.heads * DODGE[cond], d.share, yerr=d.share_ci, color=color, marker=mk, markersize=3.5,
                     linestyle=ls, linewidth=1.3, elinewidth=0.7, capsize=1.5, label=lab, zorder=3)
    axx.set_xscale("log")
    axx.set_xticks([10, 50, 200])
    axx.set_xticklabels(["10", "50", "200"], fontsize=6.5)
    axx.minorticks_off()
    axx.set_xlim(7, 290)
    axx.set_title(fmt_label, fontsize=7.5)
    axx.set_xlabel("Heads blocked", fontsize=7.5)
from matplotlib.ticker import PercentFormatter
ax_c.set_ylim(-0.12, 0.47)
ax_c.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
ax_c.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax_c.set_ylabel("Share of marker gain removed")
plt.setp(ax_c2.get_yticklabels(), visible=False)
ax_c2.legend(*ax_c.get_legend_handles_labels(), loc="upper left", frameon=False, fontsize=5.6,
             handlelength=2.0, labelspacing=0.3, borderaxespad=0.1)
ax_c.text(0, 1.13, "(c) Head blocking, with controls", transform=ax_c.transAxes, fontsize=8.5,
          ha="left", va="bottom")

# ---- Panel (d): dip mechanism -- attention to correct target (from former Fig 3a) ----
ax_d = fig.add_subplot(gs[1, 1])
retrieval = pd.read_csv(RESULTS_DIR / "dip_explanation_retrieval_attn.csv")
top10 = retrieval.groupby("k")["correct_target_attn"].apply(lambda s: s.sort_values(ascending=False).head(10).mean())
k_labels = {1: "k = 1", 2: "k = 2", 8: "k = 8"}
ks = [1, 2, 8]
colors = [CAT_BLUE, CAT_ORANGE, CAT_AQUA]
bars = ax_d.bar([k_labels[k] for k in ks], [top10[k] for k in ks], color="#0072B2", width=0.6)
for bar, k in zip(bars, ks):
    ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{top10[k]:.3f}",
               ha="center", fontsize=8)
ax_d.set_ylabel("Mean attention to correct target\n(top-10 heads)")
ax_d.set_title("(d) Attention to the needed input", fontsize=8.5)
ax_d.set_ylim(0, 0.65)

# No suptitle: the paper's caption carries the message.
OUT = REPO_ROOT / "figs" / "fig2_mechanism_merged.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Best boundary-attending head: L{l}/H{h}, score={best_score:.3f}")
