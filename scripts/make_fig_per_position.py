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


def bars(ax, x, base, within, within_ci, post, post_ci, w=0.26):
    """Three bars per model: no-marker accuracy at the same positions (gray), within-segment and
    post-marker accuracy with markers (color), 95% intervals on the marker bars."""
    ax.bar(x - w, base, w, color=BASE, alpha=0.45, linewidth=0, zorder=2)
    ax.bar(x, within, w, color=WITHIN, linewidth=0, zorder=3)
    ax.errorbar(x, within, yerr=within_ci, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=4)
    ax.bar(x + w, post, w, color=POST, linewidth=0, zorder=3)
    ax.errorbar(x + w, post, yerr=post_ci, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=4)


def make_fig1():
    from matplotlib.patches import Patch
    blocks = [(b, t) for b, t in BLOCKS if (vals.block == b).any()]
    fig, axes = plt.subplots(len(blocks), 1, figsize=(MULTI_COL_WIDTH, 1.55 * len(blocks) + 0.5), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (block, title) in zip(axes, blocks):
        for i, r in enumerate(block_means(block)):
            if r is None:
                ax.text(i, 0.5, "not run", ha="center", va="center", fontsize=6.5, color=BASE)
                continue
            # the no-marker bar shows the no-marker run at the post-marker indices (the comparison the
            # text makes); the within-segment comparison differs from it by at most 0.03 in any cell
            bars(ax, i, r["base_post"], r["within"], r["within_ci"], r["post"], r["post_ci"])
        ax.set_title(title)
        chance = 0.292 if block.startswith("Position-local") else (0.5 + 0.5 + 0.292 + 0.455) / 4
        ax.axhline(chance, color=BASE, linestyle=":", linewidth=0.9, zorder=1)
        ax.text(len(MODELS) - 0.55, chance + 0.012, "chance", fontsize=6, color=BASE, ha="right", va="bottom")
        ax.set_ylabel("Character accuracy")
        ax.set_ylim(0.2, 1.0)
    axes[-1].set_xticks(range(len(MODELS)))
    axes[-1].set_xticklabels([SHORT[m] for m in MODELS], fontsize=6.8)
    axes[-1].set_xlim(-0.6, len(MODELS) - 0.4)
    handles = [Patch(facecolor=BASE, alpha=0.45, label="No markers, same positions"),
               Patch(facecolor=WITHIN, label="Markers: within a segment"),
               Patch(facecolor=POST, label="Markers: right after a marker")]
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    fig.tight_layout()
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
                w = 0.36
                ax.bar(i - w / 2, a, w, color=color, alpha=0.35, linewidth=0, zorder=2)
                ax.errorbar(i - w / 2, a, yerr=o[f"{col}_ci"].iloc[0], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=4)
                ax.bar(i + w / 2, b, w, color=color, linewidth=0, zorder=3)
                ax.errorbar(i + w / 2, b, yerr=r[f"{col}_ci"].iloc[0], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=4)
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
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=INK, alpha=0.35, label="Dependency intact (original task)"),
               Patch(facecolor=INK, label="Dependency severed (recurrence restarts at each marker)")]
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
