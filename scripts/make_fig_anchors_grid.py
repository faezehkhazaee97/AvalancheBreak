#!/usr/bin/env python3
"""Figure 3: Tier-2/3 anchor grids (figs/anchors_grid.pdf). copy_previous
and sliding_window_max, one panel per model, TAR vs. sequence length, Base
vs. Cellular_Bounds (pooled across densities k=1/2/4/8/16, k=3 excluded).
Replaces old Figures 8-9. running_max (originally the second anchor here)
was retired: its cumulative-max statistic saturates at the alphabet
ceiling, making TAR spuriously RISE with N instead of showing the real
difficulty curve -- not a valid boundary-effect comparison point.
sliding_window_max (a bounded 3-step lookback that can't saturate the same
way) is its replacement Tier-3 anchor.

copy_previous has NO Alphabet_Size axis at all -- it's fixed-alphabet
binary (0/1), not a sweep with one value missing. sliding_window_max
sweeps Alphabet_Size in {4, 8} only; alphabet=2 is a deliberate exclusion,
not missing data (see sliding_window_max/engine.py: screened weak at
alphabet=2, model exploits the skewed output distribution via lazy
constant-guessing 65% of the time instead of genuine window-tracking).
Base is now colored to match its own series instead of plain black
regardless of alphabet -- previously sliding_window_max's two Base lines
(A=4, A=8) were both drawn in identical black, visually collapsing into
what looked like one line when it was really two overlapping ones. Random-
chance lines (1/alphabet size; 1/2 for copy_previous's fixed binary) added,
same convention as Figure 2/A1.

Both tasks' canonical files are already fragment-complete (a
_trials20-39.csv fragment exists standalone for Qwen2.5-7B, unlike
cyclic_shift), so select_canonical_files() is used directly with no
pooled-file workaround needed here.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.compute_diagnostics import select_canonical_files
from tasks.common.figure_style import (
    set_rc, savefig, AXIS_COLORS, BASE_STYLE, BOUNDS_STYLE,
    MODEL_ORDER, MODEL_LABELS, MULTI_COL_WIDTH, set_power_of_2_xticks,
)

set_rc()

TASKS = ["copy_previous", "sliding_window_max"]
# copy_previous has no Alphabet_Size axis -- fixed binary, deliberately NOT
# drawn from AXIS_COLORS (which codes actual alphabet-size values) so it
# can never be mistaken for "A=<something>".
BINARY_COLOR = "black"
base_style_no_color = {k: v for k, v in BASE_STYLE.items() if k != "color"}

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
            color = AXIS_COLORS.get(axis_val, BINARY_COLOR) if axis_val is not None else BINARY_COLOR

            ax.plot(base.index, base["mean"], color=color, **base_style_no_color)
            ax.fill_between(base.index, base["mean"] - base["sem"], base["mean"] + base["sem"],
                             color=color, alpha=0.08)
            ax.plot(bnd.index, bnd["mean"], color=color, **BOUNDS_STYLE)
            ax.fill_between(bnd.index, bnd["mean"] - bnd["sem"], bnd["mean"] + bnd["sem"],
                             color=color, alpha=0.15)

            # Random chance = 1/alphabet size (1/2 for copy_previous's fixed
            # binary alphabet), color-matched to its own series like Base/
            # Bounds above.
            chance = 1.0 / axis_val if axis_val is not None else 0.5
            ax.axhline(chance, color=color, linestyle=":", linewidth=1, alpha=0.7)

        set_power_of_2_xticks(ax, [4, 8, 16, 32, 64], fontsize=6)
        if row == 0:
            ax.set_title(MODEL_LABELS[model])
        if col == 0:
            ax.set_ylabel(f"{task}\nTAR")
        if row == 1:
            ax.set_xlabel("Sequence length N")
        ax.set_ylim(0, 1.05)

# Two-part legend, same split as Figure 1/A1: color = series identity
# (copy_previous is fixed-binary, never an alphabet value -- labeled by
# name, not folded into the A=X code), linestyle/marker = Base vs.
# Cellular_Bounds vs. random chance, shared across both rows.
color_handles = [Line2D([0], [0], color=BINARY_COLOR, linewidth=2, label="copy_previous (binary)")] + [
    Line2D([0], [0], color=AXIS_COLORS[a], linewidth=2, label=f"sliding_window_max A={int(a)}")
    for a in sorted(AXIS_COLORS.keys()) if a in (4, 8)
]
style_handles = [
    Line2D([0], [0], color="#555555", linestyle=BASE_STYLE["linestyle"], marker=BASE_STYLE.get("marker", "o"),
           markersize=4, label="Base"),
    Line2D([0], [0], color="#555555", linestyle=BOUNDS_STYLE["linestyle"], marker=BOUNDS_STYLE.get("marker", "s"),
           markersize=4, label="Cellular_Bounds"),
    Line2D([0], [0], color="#999999", linestyle=":", linewidth=1.5, label="Random chance (1/A)"),
]
fig.legend(handles=color_handles + style_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           ncol=3, frameon=False, fontsize=7)
fig.suptitle("Tier-2/3 anchors: Base (dashed) vs. Cellular_Bounds (solid), colored by series", y=1.14)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "anchors_grid.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
