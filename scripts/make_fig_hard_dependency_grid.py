#!/usr/bin/env python3
"""Figure 4: Hard-dependency tasks (figs/hard_dependency_grid.pdf).
running_xor and running_sum_local (all three moduli) -- one panel per
model per task, TAR vs. sequence length, Base vs. Cellular_Bounds (colored
solid, pooled across densities, k=3 excluded). Replaces old Figures 6-7.
Legend moved outside the axes (was sitting on top of data in the
bottom-left panel).

sliding_window_max (originally a third row here) was removed: it was
plotted in BOTH this figure and Figure A2 with identical data, and it
isn't actually arithmetic the way running_xor (XOR chain) and
running_sum_local (modular running sum) are -- it's a comparison/max
operation, so it fit this figure's "arithmetic-bearing dependency" theme
worse than it fit A2's "anchor task" theme. Kept only in A2 now; no
information lost, just de-duplicated.

running_xor has no Alphabet_Size/Modulus axis at all (fixed binary) --
previously fell through to a hardcoded "#d95f02" fallback color that
happens to be IDENTICAL to M=4's color (a coincidental collision, not by
design; unlike different tasks legitimately sharing a color for the same
axis VALUE elsewhere in this report, which figure_style.py's own
docstring documents as intentional: "one color per axis value, consistent
across figures"). running_xor's single series now gets its own distinct
BINARY_COLOR, matching Figure A2's copy_previous fix, and its legend
label (previously an empty string -- present in the legend as a blank,
unlabeled swatch) is now spelled out. Random-chance lines (1/axis value;
1/2 for running_xor's fixed binary) added, same convention as Figures
2/A1/A2.
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

TASKS = ["running_xor", "running_sum_local"]
TASK_LABELS = {"running_xor": "running_xor (binary)", "running_sum_local": "running_sum_local"}
BINARY_COLOR = "black"  # running_xor has no axis at all -- distinct from any AXIS_COLORS numeral, not a coincidental reuse

fig, axes = plt.subplots(2, 5, figsize=(MULTI_COL_WIDTH, 4.6), sharey=True)

print("Provenance -- canonical files used for Figure 4:")
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
        key_cols = [c for c in ["Method", "Interval", "Seq_Length", "Modulus", "Alphabet_Size", "Trial"] if c in df.columns]
        df = df.drop_duplicates(subset=key_cols)
        df = df[df.Interval != 3]

        axis_col = "Modulus" if "Modulus" in df.columns else ("Alphabet_Size" if "Alphabet_Size" in df.columns else None)
        has_axis = axis_col and df[axis_col].notna().any()
        axis_vals = sorted(df[axis_col].dropna().unique()) if has_axis else [None]
        label_prefix = "M" if axis_col == "Modulus" else "A"

        base_style_no_color = {k: v for k, v in BASE_STYLE.items() if k != "color"}
        for axis_val in axis_vals:
            sub = df[df[axis_col] == axis_val] if axis_val is not None else df
            base = sub[sub.Method == "Base"].groupby("Seq_Length")["TAR"].agg(["mean", "sem"])
            bnd = sub[sub.Method == "Cellular_Bounds"].groupby("Seq_Length")["TAR"].agg(["mean", "sem"])
            color = AXIS_COLORS.get(axis_val, BINARY_COLOR) if axis_val is not None else BINARY_COLOR
            label_suffix = f"{TASK_LABELS[task]} {label_prefix}={int(axis_val)}" if axis_val is not None else TASK_LABELS[task]

            # Base colored to match its own axis value (modulus/alphabet),
            # not plain black for every line in the panel -- otherwise
            # multiple Base lines overlap and become indistinguishable.
            ax.plot(base.index, base["mean"], color=color, **base_style_no_color)
            ax.fill_between(base.index, base["mean"] - base["sem"], base["mean"] + base["sem"],
                             color=color, alpha=0.08)
            ax.plot(bnd.index, bnd["mean"], color=color, **BOUNDS_STYLE,
                     label=label_suffix if col == 0 else None)
            ax.fill_between(bnd.index, bnd["mean"] - bnd["sem"], bnd["mean"] + bnd["sem"],
                             color=color, alpha=0.15)

            # Random chance = 1/axis value (1/2 for running_xor's fixed
            # binary alphabet), color-matched to its own series.
            chance = 1.0 / axis_val if axis_val is not None else 0.5
            ax.axhline(chance, color=color, linestyle=":", linewidth=1, alpha=0.7)

        set_power_of_2_xticks(ax, df.Seq_Length.unique(), fontsize=6)
        if row == 0:
            ax.set_title(MODEL_LABELS[model])
        if col == 0:
            ax.set_ylabel(f"{task}\nTAR")
        if row == 1:
            ax.set_xlabel("Sequence length N")
        ax.set_ylim(0, 1.05)

# Shared legend: color = axis value (deduped across M=.../A=... labels
# that reuse the same color), plus one neutral-gray Base/Bounds linestyle
# pair -- avoids both "legend only in first panel" and "Base is
# indistinguishable black for every color" at once.
color_handles, color_labels = [], []
for row in range(len(TASKS)):
    h, l = axes[row, 0].get_legend_handles_labels()
    for hi, li in zip(h, l):
        if li not in color_labels:
            color_handles.append(hi)
            color_labels.append(li)
style_handles = [
    Line2D([0], [0], color="gray", linestyle=BASE_STYLE["linestyle"], marker=BASE_STYLE.get("marker", "o"),
           markersize=4, label="Base"),
    Line2D([0], [0], color="gray", linestyle=BOUNDS_STYLE["linestyle"], marker=BOUNDS_STYLE.get("marker", "s"),
           markersize=4, label="Cellular_Bounds"),
    Line2D([0], [0], color="#999999", linestyle=":", linewidth=1.5, label="Random chance (1/axis value)"),
]
fig.legend(style_handles + color_handles, [h.get_label() for h in style_handles] + color_labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=6.5)
fig.suptitle("Hard-dependency tasks: Base (dashed) vs. Cellular_Bounds (solid), colored by series", y=1.09)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "hard_dependency_grid.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
