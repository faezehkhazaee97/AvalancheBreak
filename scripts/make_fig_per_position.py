#!/usr/bin/env python3
"""Figure 1: per-position accuracy, every model.

For each block of tasks (position-local with letters, position-local with digits once
those runs exist, and dependency tasks) and each model: accuracy at the position right
after a marker and at positions within a segment, each next to the no-marker (Base)
accuracy at the same position indices. Mean over the tasks in the block; whiskers are
95% confidence intervals (combined across tasks). Reads
results_clean/_summary/per_position_values.csv; run scripts/analyze_per_position.py first.

Also writes the reset figure (fig_reset_per_position): for the four dependency tasks,
post-marker and within-segment accuracy with the dependency intact vs. severed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH  # noqa: E402

POST, WITHIN, BASE, INK = "#0072B2", "#D55E00", "#6e6e6e", "#1a1a1a"
MODELS = ["Qwen2.5 7B", "Qwen2.5 14B", "Qwen2.5 32B", "Mistral 7B", "Phi-3.5 mini",
          "Qwen3.8 27B", "Gemma 4 31B", "Gemini 3.5 Flash", "Gemini 3.6 Flash"]
SHORT = {m: m.replace(" ", "\n", 1) for m in MODELS}
BLOCKS = [
    ("Position-local, letters", "Position-local tasks, letters (merge into multi-letter tokens without markers)"),
    ("Position-local, spaced letters", "Position-local tasks, spaced letters (one token per letter either way)"),
    ("Dependency, as is", "Dependency tasks (digits, one token per digit either way)"),
]

vals = pd.read_csv(REPO / "results_clean" / "_summary" / "per_position_values.csv")


def block_means(block):
    s = vals[vals.block == block]
    out = []
    for m in MODELS:
        x = s[s.model == m]
        if x.empty:
            out.append(None)
            continue
        n = len(x)
        out.append(dict(
            base_post=x.base_at_post.mean(), post=x.post.mean(),
            post_ci=np.sqrt((x.post_ci ** 2).sum()) / n,
            base_within=x.base_at_within.mean(), within=x.within.mean(),
            within_ci=np.sqrt((x.within_ci ** 2).sum()) / n,
        ))
    return out


def make_fig1():
    """Forest plot: one panel per block; rows are models; for each, the change with markers at the
    within-segment positions (orange circle) and at the position right after a marker (blue diamond),
    each against the no-marker run at the same positions, with 95% intervals; zero line."""
    from matplotlib.lines import Line2D
    blocks = [(b, t) for b, t in BLOCKS if (vals.block == b).any()]
    fig, axes = plt.subplots(1, len(blocks), figsize=(MULTI_COL_WIDTH, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    n = len(MODELS)
    for ax, (block, title) in zip(axes, blocks):
        ax.axvline(0, color=BASE, linewidth=0.7, zorder=1)
        for i, r in enumerate(block_means(block)):
            y = n - 1 - i
            if r is None:
                ax.text(0, y, "not run", ha="center", va="center", fontsize=6, color=BASE)
                continue
            ax.errorbar(r["within"] - r["base_within"], y + 0.17, xerr=r["within_ci"], fmt="o", markersize=3.4, color=WITHIN,
                        ecolor=WITHIN, elinewidth=0.8, capsize=1.5, zorder=3)
            ax.errorbar(r["post"] - r["base_post"], y - 0.17, xerr=r["post_ci"], fmt="D", markersize=3.2, color=POST,
                        ecolor=POST, elinewidth=0.8, capsize=1.5, zorder=3)
        ax.set_title(title.split(" (")[0], fontsize=7.5)
        ax.set_xlabel("Change in accuracy with markers")
        ax.set_xlim(-0.55, 0.55)
        ax.grid(True, axis="x", color="#e6e6e6", linewidth=0.5); ax.grid(False, axis="y")
        for yy in range(n):
            ax.axhline(yy - 0.5, color="#f0f0f0", linewidth=0.5, zorder=0)
    axes[0].set_yticks(range(n)); axes[0].set_yticklabels([SHORT[m].replace("\n", " ") for m in MODELS][::-1], fontsize=6.8)
    axes[0].set_ylim(-0.6, n - 0.4)
    handles = [Line2D([], [], marker="o", linestyle="none", color=WITHIN, markersize=4, label="Within a segment"),
               Line2D([], [], marker="D", linestyle="none", color=POST, markersize=3.8, label="Right after a marker")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = REPO / "figs" / "fig1_per_position.pdf"
    savefig(fig, out)
    fig.savefig(REPO / "figs" / "png" / "fig1_per_position.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


def make_reset_fig():
    tasks = ["running_xor", "running_sum_local", "sliding_window_max", "copy_previous"]
    models = ["Qwen2.5 7B", "Qwen2.5 32B", "Qwen3.8 27B", "Gemma 4 31B"]
    fig, axes = plt.subplots(2, len(tasks), figsize=(MULTI_COL_WIDTH, 3.6), sharey=True)
    for j, t in enumerate(tasks):
        for row, (col, color, label) in enumerate([("post", POST, "Right after a marker"),
                                                   ("within", WITHIN, "Within a segment")]):
            ax = axes[row, j]
            for i, m in enumerate(models):
                o = vals[(vals.task == t) & (vals.model == m)]
                r = vals[(vals.task == f"{t}_reset") & (vals.model == m)]
                if o.empty or r.empty:
                    ax.text(i, 0.6, "not\nrun", ha="center", va="center", fontsize=5.5, color=BASE, linespacing=0.9)
                    continue
                a, b = o[col].iloc[0], r[col].iloc[0]
                ax.annotate("", xy=(i, b), xytext=(i, a), arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.1, mutation_scale=7), zorder=3)
                ax.plot(i, a, marker="o", markersize=3, color=color, markerfacecolor="white", markeredgewidth=1.0, zorder=4)
                ax.errorbar(i, b, yerr=r[f"{col}_ci"].iloc[0], fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, zorder=4)
            ax.set_ylim(0.15, 1.05)
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_xticks(range(len(models)))
            ax.set_xlim(-0.6, len(models) - 0.4)
            ax.set_xticklabels(["Q2.5\n7B", "Q2.5\n32B", "Q3.8\n27B", "Gemma\n4"], fontsize=6.0)
            if row == 0:
                ax.set_title(t, fontsize=7.5, family="monospace")
            if j == 0:
                ax.set_ylabel(f"{label}\ntoken accuracy", fontsize=7.5)
    handles = [Line2D([], [], marker="o", linestyle="none", color=INK, markerfacecolor="white", markeredgewidth=1.0, markersize=4, label="Dependency intact (original task)"),
               Line2D([], [], marker=">", linestyle="-", color=INK, markersize=4, label="Dependency severed (recurrence restarts at each marker)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.8)
    fig.tight_layout()
    out = REPO / "figs" / "fig_reset_per_position.pdf"
    savefig(fig, out)
    fig.savefig(REPO / "figs" / "png" / "fig_reset_per_position.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    set_rc()
    make_fig1()
    make_reset_fig()
