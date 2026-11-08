#!/usr/bin/env python3
"""Missing Phi-3.5-mini-instruct panel for the running_sum_local by-model
appendix figure -- matches the existing 4-model panel style exactly
(TAR top row / SAR bottom row, one column per N, colored by modulus,
Base plotted as a star at density=0, dotted chance lines).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = REPO_ROOT / "results_clean" / "running_sum_local" / "grid"
OUT_DIR = REPO_ROOT / "figs" / "png"

LENGTHS = [4, 8, 16, 32, 64]
MODULI = [2, 4, 8]
COLORS = {2: "#2166ac", 4: "#238b45", 8: "#d6604d"}
MARKERS = {2: "o", 4: "s", 8: "^"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
})

df = pd.read_csv(GRID_DIR / "Phi-3.5-mini-instruct.csv")
df = df[df.Interval != 3]

fig, axes = plt.subplots(2, len(LENGTHS), figsize=(14, 4.6), sharex=True)

base_means = {}
for m in MODULI:
    base_means[m] = df[(df.Modulus == m) & (df.Method == "Base")]["TAR"].mean()

for col, N in enumerate(LENGTHS):
    for row, metric in enumerate(["TAR", "SAR"]):
        ax = axes[row, col]
        for m in MODULI:
            color = COLORS[m]
            marker = MARKERS[m]
            sub = df[(df.Modulus == m) & (df.Seq_Length == N)]

            base = sub[sub.Method == "Base"][metric]
            if len(base):
                ax.errorbar(0, base.mean(), yerr=base.sem(), color=color,
                            marker="*", markersize=10, capsize=2, linestyle="none", zorder=5)

            bnd = sub[sub.Method == "Cellular_Bounds"]
            if len(bnd):
                g = bnd.groupby("Interval")[metric].agg(["mean", "sem"]).sort_index()
                density = 1.0 / g.index.values
                order = np.argsort(density)
                ax.errorbar(density[order], g["mean"].values[order], yerr=g["sem"].values[order],
                            color=color, marker=marker, markersize=4, linewidth=1.5, capsize=2,
                            label=f"Modulus={m} (Base: {base_means[m]:.3f})" if (row == 0 and col == 0) else None)

            chance = 1.0 / m
            ax.axhline(chance, color=color, linestyle=":", linewidth=1, alpha=0.6)

        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(f"{metric}, N={N}", fontsize=8)
        if row == 1:
            ax.set_xlabel("Boundary Density (1/k)")
        if col == 0:
            ax.set_ylabel(metric)

axes[0, 0].legend(loc="upper left", frameon=False, fontsize=6)
fig.suptitle(
    "running_sum_local (all moduli) -- Phi-3.5-mini (20 trials/point)\n"
    "Stars = Base (no boundaries), per modulus. Dotted lines = that modulus's random chance. Modulus=2 is the XOR-equivalent case.",
    fontsize=8.5, y=1.03,
)
fig.tight_layout()

out = OUT_DIR / "running_sum_local_Phi-3.5-mini-instruct.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out}")
