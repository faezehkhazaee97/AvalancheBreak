#!/usr/bin/env python3
"""Figure: sliding_window_max grid (figs/sliding_window_max.pdf) -- the
Tier-3 anchor replacement for running_max. One panel per model, alphabet
sizes {4,8} (alphabet=2 dropped: screened weak, model exploits the skewed
output distribution via lazy constant-guessing instead of genuine
tracking -- see tasks/sliding_window_max/engine.py). Base (black dashed)
vs. Cellular_Bounds (colored solid, pooled across densities, k=3 excluded).
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import (
    set_rc, savefig, AXIS_COLORS, BASE_STYLE, BOUNDS_STYLE,
    MODEL_ORDER, MODEL_LABELS, MULTI_COL_WIDTH, set_power_of_2_xticks,
)

set_rc()

GRID_DIR = REPO_ROOT / "results_clean" / "sliding_window_max" / "grid"
print(f"Provenance: {GRID_DIR.relative_to(REPO_ROOT)}/*.csv")

fig, axes = plt.subplots(1, 5, figsize=(MULTI_COL_WIDTH, 2.6), sharey=True)

for col, model in enumerate(MODEL_ORDER):
    ax = axes[col]
    f = GRID_DIR / f"{model}.csv"
    df = pd.read_csv(f)
    df = df[df.Interval != 3]

    for axis_val in sorted(df.Alphabet_Size.unique()):
        sub = df[df.Alphabet_Size == axis_val]
        base = sub[sub.Method == "Base"].groupby("Seq_Length")["TAR"].agg(["mean", "sem"])
        bnd = sub[sub.Method == "Cellular_Bounds"].groupby("Seq_Length")["TAR"].agg(["mean", "sem"])
        color = AXIS_COLORS.get(axis_val, "gray")

        ax.plot(base.index, base["mean"], **BASE_STYLE, label="Base" if (axis_val == 4 and col == 0) else None)
        ax.fill_between(base.index, base["mean"] - base["sem"], base["mean"] + base["sem"], color="black", alpha=0.08)
        ax.plot(bnd.index, bnd["mean"], color=color, **BOUNDS_STYLE, label=f"A={int(axis_val)}" if col == 0 else None)
        ax.fill_between(bnd.index, bnd["mean"] - bnd["sem"], bnd["mean"] + bnd["sem"], color=color, alpha=0.15)

    set_power_of_2_xticks(ax, df.Seq_Length.unique(), fontsize=6)
    ax.set_title(MODEL_LABELS[model], fontsize=8)
    ax.set_xlabel("Sequence length N")
    if col == 0:
        ax.set_ylabel("TAR")
        ax.legend(loc="lower left", frameon=False, fontsize=6)
    ax.set_ylim(0, 1.0)

fig.suptitle("sliding_window_max (window=3): Base (black dashed) vs. Cellular_Bounds (colored solid)", y=1.05, fontsize=8)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "sliding_window_max.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
