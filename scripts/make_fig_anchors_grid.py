#!/usr/bin/env python3
"""Figure 3: Tier-2/3 anchor grids (figs/anchors_grid.pdf). copy_previous
and sliding_window_max, one panel per model, all alphabet sizes, TAR vs.
sequence length, Base (black dashed) vs. Cellular_Bounds (colored solid,
pooled across densities k=1/2/4/8/16, k=3 excluded). Replaces old Figures
8-9. running_max (originally the second anchor here) was retired: its
cumulative-max statistic saturates at the alphabet ceiling, making TAR
spuriously RISE with N instead of showing the real difficulty curve --
not a valid boundary-effect comparison point. sliding_window_max (a
bounded 3-step lookback that can't saturate the same way) is its
replacement Tier-3 anchor.

Both tasks' canonical files are already fragment-complete (a
_trials20-39.csv fragment exists standalone for Qwen2.5-7B, unlike
cyclic_shift), so select_canonical_files() is used directly with no
pooled-file workaround needed here.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.compute_diagnostics import select_canonical_files
from tasks.common.figure_style import (
    set_rc, savefig, AXIS_COLORS, BASE_STYLE, BOUNDS_STYLE,
    MODEL_ORDER, MODEL_LABELS, MULTI_COL_WIDTH, set_power_of_2_xticks,
)

set_rc()

TASKS = ["copy_previous", "sliding_window_max"]

fig, axes = plt.subplots(2, 5, figsize=(MULTI_COL_WIDTH, 4.6), sharey=True)

print("Provenance -- canonical files used for Figure 3:")
for row, task in enumerate(TASKS):
    model_files = select_canonical_files(task)
    print(f"  {task}:")
    for m, paths in model_files.items():
        print(f"    {m}: {[str(p.relative_to(REPO_ROOT)) for p in paths]}")

    for col, model in enumerate(MODEL_ORDER):
        ax = axes[row, col]
        if model not in model_files:
            ax.axis("off")
            continue
        frames = [pd.read_csv(p) for p in model_files[model]]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        key_cols = [c for c in ["Method", "Interval", "Seq_Length", "Alphabet_Size", "Trial"] if c in df.columns]
        df = df.drop_duplicates(subset=key_cols)
        df = df[df.Interval != 3]

        has_axis = "Alphabet_Size" in df.columns and df.Alphabet_Size.notna().any()
        axis_vals = sorted(df.Alphabet_Size.dropna().unique()) if has_axis else [None]

        for axis_val in axis_vals:
            sub = df[df.Alphabet_Size == axis_val] if axis_val is not None else df
            base = sub[sub.Method == "Base"].groupby("Seq_Length")["TAR"].agg(["mean", "sem"])
            bnd = sub[sub.Method == "Cellular_Bounds"].groupby("Seq_Length")["TAR"].agg(["mean", "sem"])
            color = AXIS_COLORS.get(axis_val, "#333333") if axis_val is not None else "#d95f02"
            label_suffix = f"A={int(axis_val)}" if axis_val is not None else ""

            ax.plot(base.index, base["mean"], **BASE_STYLE,
                     label="Base" if (axis_val in (2, None) and row == 0 and col == 0) else None)
            ax.fill_between(base.index, base["mean"] - base["sem"], base["mean"] + base["sem"],
                             color="black", alpha=0.08)
            ax.plot(bnd.index, bnd["mean"], color=color, **BOUNDS_STYLE,
                     label=label_suffix if (row == 1 and col == 0) else None)
            ax.fill_between(bnd.index, bnd["mean"] - bnd["sem"], bnd["mean"] + bnd["sem"],
                             color=color, alpha=0.15)

        set_power_of_2_xticks(ax, [4, 8, 16, 32, 64], fontsize=6)
        if row == 0:
            ax.set_title(MODEL_LABELS[model])
        if col == 0:
            ax.set_ylabel(f"{task}\nTAR")
        if row == 1:
            ax.set_xlabel("Sequence length N")
        ax.set_ylim(0, 1.05)

# One shared legend for the whole figure (was sitting only in the
# top-left panel, but applies to every panel) -- moved outside the axes.
h0, l0 = axes[0, 0].get_legend_handles_labels()
h1, l1 = axes[1, 0].get_legend_handles_labels()
all_handles, all_labels = [], []
for h, l in zip(h0 + h1, l0 + l1):
    if l not in all_labels:
        all_handles.append(h)
        all_labels.append(l)
fig.legend(all_handles, all_labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           ncol=len(all_labels), frameon=False, fontsize=7)
fig.suptitle("Tier-2/3 anchors: Base (black dashed) vs. Cellular_Bounds (colored solid)", y=1.07)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "anchors_grid.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
