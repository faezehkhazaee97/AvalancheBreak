#!/usr/bin/env python3
"""Missing Phi-3.5-mini-instruct panel for the copy_previous by-model
appendix figure -- matches the existing 4-model panel style (TAR top row /
SAR bottom row, one column per N, black star = Base, gray line = random
chance (0.5, binary task), one color per N-panel for the Cellular_Bounds
density sweep).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = REPO_ROOT / "results_clean" / "copy_previous" / "grid"
OUT_DIR = REPO_ROOT / "figs" / "png"

LENGTHS = [4, 8, 16, 32, 64]
COLORS = ["#2a78d6", "#238b45", "#d6604d", "#eb8c34", "#1baf9e"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
})

df = pd.read_csv(GRID_DIR / "Phi-3.5-mini-instruct.csv")
df = df[df.Interval != 3]

fig, axes = plt.subplots(2, len(LENGTHS), figsize=(14, 4.6), sharex=True)

for col, N in enumerate(LENGTHS):
    color = COLORS[col % len(COLORS)]
    for row, metric in enumerate(["TAR", "SAR"]):
        ax = axes[row, col]
        sub = df[df.Seq_Length == N]

        ax.axhline(0.5, color="gray", linestyle="-", linewidth=1, alpha=0.7,
                   label="Random chance (0.5)" if (row == 0 and col == 0) else None)

        base = sub[sub.Method == "Base"][metric]
        if len(base):
            ax.errorbar(0, base.mean(), yerr=base.sem(), color="black",
                        marker="*", markersize=12, capsize=2, linestyle="none", zorder=5,
                        label=f"Base / no boundaries ({base.mean():.3f})" if (row == 0 and col == 0) else None)

        bnd = sub[sub.Method == "Cellular_Bounds"]
        if len(bnd):
            g = bnd.groupby("Interval")[metric].agg(["mean", "sem"]).sort_index()
            density = 1.0 / g.index.values
            order = np.argsort(density)
            ax.errorbar(density[order], g["mean"].values[order], yerr=g["sem"].values[order],
                        color=color, marker="o", markersize=5, linewidth=1.5, capsize=2,
                        label="With boundaries" if (row == 0 and col == 0) else None)

        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(f"{metric}, N={N}", fontsize=8)
        if row == 1:
            ax.set_xlabel(f"Boundary Density (1/k)")
        if col == 0:
            ax.set_ylabel(metric)

axes[0, 0].legend(loc="lower left", frameon=False, fontsize=6)
fig.suptitle(
    "copy_previous -- Phi-3.5-mini (single 40-trial batch)\n"
    "Black star = Base (no boundaries). Colored dots = with boundaries at that density.",
    fontsize=8.5, y=1.03,
)
fig.tight_layout()

out = OUT_DIR / "copy_previous_Phi-3.5-mini-instruct.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out}")
