#!/usr/bin/env python3
"""New appendix figure: does a fitted quadratic curve survive a held-out
test, by tier? Visualizes the "Dose-response fit robustness" table's key
finding (Qwen2.5-32B-Instruct, includes N=128) -- currently table-only,
no figure. Bar chart: fraction of density-response curves where a
quadratic beats a plain straight line on LEFT-OUT predictions (not just
in-sample fit). Tier-1 tasks clear 42-50%; every Tier-2/3 task is 0%.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH

set_rc()

# From results_clean/_dose_response_robustness/dose_response_robustness_32b.csv
DATA = [
    ("cyclic_shift", 1, 47),
    ("random_permutation", 1, 42),
    ("conditional_shift", 1, 50),
    ("copy_previous", 2, 0),
    ("running_xor", 2, 0),
    ("running_sum_local", 2, 0),
    ("sliding_window_max", 3, 0),
]

CAT_BLUE = "#2a78d6"
CAT_ORANGE = "#eb6834"
CAT_AQUA = "#1baf7a"
TIER_COLOR = {1: CAT_BLUE, 2: CAT_ORANGE, 3: CAT_AQUA}

fig, ax = plt.subplots(figsize=(MULTI_COL_WIDTH * 0.65, 3.2))

labels = [d[0] for d in DATA]
values = [d[2] for d in DATA]
colors = [TIER_COLOR[d[1]] for d in DATA]

bars = ax.bar(labels, values, color=colors, width=0.6)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v}%", ha="center", fontsize=8)

ax.set_ylabel("% of curves where quadratic beats linear\non held-out (leave-one-density-out) predictions")
ax.set_ylim(0, 60)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
ax.set_title("Does a fitted curve actually survive a held-out test?\n(Qwen2.5-32B-Instruct, includes N=128)", fontsize=9)

from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=CAT_BLUE, label="Tier 1 (no dependency)"),
    Patch(facecolor=CAT_ORANGE, label="Tier 2 (one-step dependency)"),
    Patch(facecolor=CAT_AQUA, label="Tier 3 (bounded window)"),
]
ax.legend(handles=legend_elems, loc="upper right", frameon=False, fontsize=7)

fig.tight_layout()

OUT = REPO_ROOT / "figs" / "cv_robustness.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
