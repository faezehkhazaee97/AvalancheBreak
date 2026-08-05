#!/usr/bin/env python3
"""Rebuilds the report's introductory 'Boundary effect by dependency
locality' overview chart (all tasks, Qwen2.5-7B-Instruct). The original
generator script no longer exists in the repo; this rebuild uses the same
canonical per-task deltas/sigmas (re-derived directly from results_clean,
matching the values already visible in the old chart) with ONE change:
running_max is replaced by sliding_window_max, since running_max was
retired this session (its cumulative-max statistic saturates and makes
TAR spuriously rise with N -- not a valid "boundary effect" comparison
point). sliding_window_max is running_max's replacement Tier-3 anchor.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH

set_rc()

# (task, delta, sigma, x_position, y_label_offset) -- re-derived from
# results_clean/<task>/grid/Qwen2.5-7B-Instruct*.csv, Interval!=3, same
# convention as every other significance number in this report.
DATA = [
    ("composed_permutation", -0.0483, -7.12, 0),
    ("cyclic_shift",          0.0978,  6.89, 1),
    ("random_permutation",    0.1109,  9.18, 1.5),
    ("character_substitution",0.1155,  9.27, 2),
    ("conditional_shift",     0.0216,  1.89, 1),
    ("running_sum_local",    -0.0302, -3.79, 3),
    ("running_xor",          -0.0092, -0.65, 4),
    ("sliding_window_max",   -0.0920, -4.61, 5),
    ("copy_previous",        -0.1120, -7.38, 3),
]

fig, ax = plt.subplots(figsize=(MULTI_COL_WIDTH, 5.5))

for task, delta, sigma, x in DATA:
    if sigma > 2:
        color = "#1b7837"  # helps
    elif sigma < -2:
        color = "#b2182b"  # hurts
    else:
        color = "#666666"  # not significant
    # SEM implied by delta/sigma (sigma = delta / combined_sem)
    sem = abs(delta / sigma) if sigma != 0 else 0
    ax.errorbar([x], [delta], yerr=[sem], fmt="o", color=color, markersize=10,
                capsize=4, linewidth=1.5, zorder=3)
    va = "bottom" if delta >= 0 else "top"
    offset = 0.008 if delta >= 0 else -0.008
    ax.annotate(task, (x, delta + offset), ha="center", va=va, fontsize=8)

ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xlim(-0.7, 5.7)
ax.set_xticks([])
ax.set_ylabel("Boundary effect size (Δ TAR = Cellular_Bounds − Base)")
ax.set_title("Boundary effect by dependency locality — Qwen2.5-7B-Instruct, all 9 tasks", fontsize=11)

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1b7837", markersize=9, label="Helps (>2σ)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#b2182b", markersize=9, label="Hurts (>2σ)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666", markersize=9, label="Not significant"),
]
ax.legend(handles=legend_elems, loc="upper right", frameon=True, fontsize=8)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "intro_dependency_locality.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
