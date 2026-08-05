#!/usr/bin/env python3
"""Figure 3 (main paper): why the misaligned boundary density hurts, shown
two ways. Panel (a) mechanistic: mean attention-to-the-actual-needed-token
(top-10 heads), copy_previous, Qwen2.5-32B, at k=1 (regular offset) vs.
k=2 (the dip) vs. k=8 (recovered) -- direct inside-the-model evidence that
retrieval of the correct target collapses specifically at the misaligned
density. Panel (b) causal: Base->Bounds delta for four original
dependency tasks next to their reset-variant delta (where the boundary
genuinely severs the dependency instead of just visually interrupting
it) -- if boundaries hurt by breaking a needed link, severing that link
on purpose should shrink or reverse the harm. copy_previous is flagged
as the one exception (rule-switching side effect, not a failure of the
causal story -- see script/HTML text for the error-position evidence).
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

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH, 3.2))

# Validated palette (dataviz skill, references/palette.md)
CAT_BLUE = "#2a78d6"    # slot 1
CAT_ORANGE = "#eb6834"  # slot 2
CAT_AQUA = "#1baf7a"    # slot 3
GRAY = "#78776f"        # text-secondary-adjacent neutral, for the "prior/original" series

# --- Panel (a): dip mechanism ---
retrieval = pd.read_csv(REPO_ROOT / "results" / "Qwen2.5-32B-Instruct" / "dip_explanation_retrieval_attn.csv")
top10 = retrieval.groupby("k")["correct_target_attn"].apply(lambda s: s.sort_values(ascending=False).head(10).mean())
k_labels = {1: "k=1\n(regular)", 2: "k=2\n(the dip)", 8: "k=8\n(recovered)"}
ks = [1, 2, 8]
colors = [CAT_BLUE, CAT_ORANGE, CAT_AQUA]
bars = ax_a.bar([k_labels[k] for k in ks], [top10[k] for k in ks], color=colors, width=0.6)
for bar, k in zip(bars, ks):
    ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{top10[k]:.3f}",
               ha="center", fontsize=8)
ax_a.set_ylabel("Mean attention to correct target\n(top-10 heads)")
ax_a.set_title("(a) Attention to the needed token collapses\nat the misaligned density\n(copy_previous, N=32)", fontsize=8.5)
ax_a.set_ylim(0, 0.65)

# --- Panel (b): causal reset-ablation ---
tasks_deltas = {
    "running_xor": (0.015, 0.126),
    "running_sum_local": (0.011, 0.110),
    "reverse_xor": (-0.061, 0.038),
    "sliding_window_max": (0.043, 0.112),
    "copy_previous": (-0.043, -0.107),
}
labels = list(tasks_deltas.keys())
orig = [tasks_deltas[t][0] for t in labels]
reset = [tasks_deltas[t][1] for t in labels]

x = np.arange(len(labels))
width = 0.35
ax_b.bar(x - width / 2, orig, width, label="Original (visual only)", color=GRAY)
ax_b.bar(x + width / 2, reset, width, label="Reset (dependency severed)", color=CAT_ORANGE)
ax_b.axhline(0, color="black", linewidth=0.8)
ax_b.set_xticks(x)
ax_b.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
ax_b.set_ylabel("Accuracy change from adding boundaries\n(TAR delta: positive = helps, negative = hurts)", fontsize=7.5)
ax_b.set_title("(b) Severing the dependency shrinks or reverses\nthe harm -- except copy_previous\n(rule-switching side effect, see text)", fontsize=8.5)
ax_b.legend(loc="upper right", frameon=False, fontsize=6.5)

fig.suptitle("Two lines of evidence for the same mechanism (Qwen2.5-32B-Instruct)", fontsize=9, y=1.03)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "fig3_dip_mechanism_and_causal.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
