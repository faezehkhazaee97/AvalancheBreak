#!/usr/bin/env python3
"""Figure 2 (main paper): the optimal boundary density shifts from dense
(short N) to moderate/sparse (long N) -- shown identically across all
three tiers, Qwen2.5-32B-Instruct only. One row per tier (cyclic_shift /
copy_previous / sliding_window_max), one panel per representative N,
TAR vs. boundary density with the empirical best point marked with a star.
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

MODEL = "Qwen2.5-32B-Instruct"

# Validated palette (dataviz skill, references/palette.md)
CAT_ORANGE = "#eb6834"  # slot 2 -- the Cellular_Bounds line
CAT_BLUE = "#2a78d6"    # slot 1 -- the best-density star


def load_task(task, axis_val_col, axis_val, n128_path=None):
    grid_dir = REPO_ROOT / "results_clean" / task / "grid"
    frames = []
    for p in grid_dir.glob(f"{MODEL}*.csv"):
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    key_cols = [c for c in ["Method", "Interval", "Seq_Length", axis_val_col, "Trial"] if c and c in df.columns]
    df = df.drop_duplicates(subset=key_cols)
    df = df[df.Interval != 3]
    if axis_val_col:
        df = df[df[axis_val_col] == axis_val]
    return df


COMMON_N = [4, 8, 16, 32, 64, 128]
TASKS = [
    ("cyclic_shift", "Alphabet_Size", 8, COMMON_N),
    ("copy_previous", None, None, COMMON_N),
    ("sliding_window_max", "Alphabet_Size", 8, COMMON_N),  # matches cyclic_shift's alphabet=8; copy_previous is binary-only, no match possible
]

fig, axes = plt.subplots(3, 6, figsize=(MULTI_COL_WIDTH * 1.4, 6.0))

for row, (task, axis_col, axis_val, lengths) in enumerate(TASKS):
    df = load_task(task, axis_col, axis_val)
    for col in range(6):
        ax = axes[row, col]
        if col >= len(lengths):
            ax.axis("off")
            continue
        N = lengths[col]
        sub = df[df.Seq_Length == N]
        base = sub[sub.Method == "Base"]["TAR"]
        bnd = sub[sub.Method == "Cellular_Bounds"].groupby("Interval")["TAR"].agg(["mean", "sem"])
        if len(base):
            ax.axhline(base.mean(), color="black", linestyle="--", linewidth=1, alpha=0.6, label="Base" if (row == 0 and col == 0) else None)
        if len(bnd):
            density = 1.0 / bnd.index.values
            order = np.argsort(density)
            ax.errorbar(density[order], bnd["mean"].values[order], yerr=bnd["sem"].values[order],
                        color=CAT_ORANGE, marker="o", markersize=4, linewidth=1.5, capsize=2)
            best_idx = bnd["mean"].values[order].argmax()
            ax.scatter([density[order][best_idx]], [bnd["mean"].values[order][best_idx]],
                      marker="*", s=140, color=CAT_BLUE, zorder=5, edgecolor="black", linewidth=0.5)
            ax.set_xscale("log", base=2)
            ax.set_xlim(0.045, 1.3)
            ax.set_xticks(density[order])
            ax.set_xticklabels([f"{d:.2g}" for d in density[order]], fontsize=6)
            ax.minorticks_off()
        ax.set_ylim(0, 1.05)
        ax.set_title(f"N={N}", fontsize=8)
        if row == 2:
            ax.set_xlabel("Boundary density (1/k)")
        if col == 0:
            ax.set_ylabel(f"{task}\nTAR", fontsize=7.5)

axes[0, 0].legend(loc="lower left", frameon=False, fontsize=6)
fig.suptitle("The optimal boundary density shifts from dense to moderate/sparse as sequences get longer\n"
             "(Qwen2.5-32B-Instruct; stars mark the empirically best density at each length)", fontsize=9, y=1.02)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "fig2_vertex_migration.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
