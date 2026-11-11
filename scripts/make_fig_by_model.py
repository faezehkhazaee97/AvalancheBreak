#!/usr/bin/env python3
"""Appendix figures: accuracy by sequence length for every open model and task.

fig_by_model_local.pdf       rows = the three position-local tasks (contiguous letters), columns =
                             the seven open models, one color per alphabet size.
fig_by_model_dependency.pdf  rows = the four dependency tasks (digits), columns = models, one
                             color per modulus or alphabet size (black: binary).
Dashed: no markers. Solid: markers, mean over spacings k in {1,2,4,8,16} with k < N.
Dotted: chance (1 / alphabet size). Bands: +/- 1 standard error. N <= 64.

These replace the earlier per-model figures (tier1_cyclic_shift, anchors_grid,
hard_dependency_grid), which covered five models and three of the seven tasks.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS  # noqa: E402
from tasks.common.figure_style import set_rc, savefig, AXIS_COLORS, MULTI_COL_WIDTH  # noqa: E402

LENGTHS = [4, 8, 16, 32, 64]
KS = [1, 2, 4, 8, 16]
GRID = "#e4e3df"


def panel(ax, d):
    axis = next((c for c in ("Alphabet_Size", "Modulus") if c in d.columns), None)
    d = d.assign(N=d.Seq_Length.astype(int), k=d.Interval.astype(int), TAR=d.TAR.astype(float))
    d = d[d.N.isin(LENGTHS) & d.k.isin([0] + KS)]
    vals = sorted(d[axis].astype(int).unique()) if axis else [None]
    for v in vals:
        x = d[d[axis].astype(int) == v] if axis else d
        color = AXIS_COLORS.get(v, "#333333") if v is not None else "#333333"
        base = x[x.Method == "Base"].groupby("N").TAR.agg(["mean", "sem"])
        bnd = x[x.Method == "Cellular_Bounds"].groupby("N").TAR.agg(["mean", "sem"])
        ax.plot(base.index, base["mean"], color=color, linestyle="--", linewidth=1.1, marker="o", markersize=2.2)
        ax.plot(bnd.index, bnd["mean"], color=color, linestyle="-", linewidth=1.4, marker="s", markersize=2.2)
        ax.fill_between(bnd.index, bnd["mean"] - bnd["sem"], bnd["mean"] + bnd["sem"], color=color, alpha=0.15,
                        linewidth=0)
        ax.axhline(1.0 / (v or 2), color=color, linestyle=":", linewidth=0.8, alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENGTHS)
    ax.set_xticklabels(["4", "", "16", "", "64"], fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make(tasks, out, color_label, color_vals):
    models = [(lab, tag) for lab, tag in MODELS if tag is not None]
    fig, axes = plt.subplots(len(tasks), len(models), figsize=(MULTI_COL_WIDTH, 1.12 * len(tasks) + 0.55),
                             sharex=True, sharey=True)
    for i, t in enumerate(tasks):
        for j, (lab, tag) in enumerate(models):
            ax = axes[i, j]
            d = load(t, tag)
            if d is None:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center", fontsize=6.5,
                        color="#78776f")
            else:
                panel(ax, d)
            if i == 0:
                ax.set_title(lab, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(t.replace("_", "_\n", 1) if len(t) > 14 else t, fontsize=6.5, family="monospace")
            if i == len(tasks) - 1:
                ax.set_xlabel("$N$", fontsize=7.5, labelpad=1)
    handles = [Line2D([], [], color="#333333", linestyle="--", marker="o", markersize=3, linewidth=1.1, label="No markers"),
               Line2D([], [], color="#333333", linestyle="-", marker="s", markersize=3, linewidth=1.4, label="Markers"),
               Line2D([], [], color="#333333", linestyle=":", linewidth=0.9, label="Chance")]
    handles += [Line2D([], [], color=AXIS_COLORS.get(v, "#333333") if v else "#333333", linewidth=2,
                       label=("binary" if v is None else f"{color_label} = {v}")) for v in color_vals]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False,
               fontsize=7)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    savefig(fig, REPO / "figs" / f"{out}.pdf")
    fig.savefig(REPO / "figs" / "png" / f"{out}.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    set_rc()
    make(["cyclic_shift", "random_permutation", "conditional_shift"], "fig_by_model_local", "$A$", [2, 4, 8])
    make(["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"], "fig_by_model_dependency",
         "$A$ or $M$", [None, 2, 4, 8])
