#!/usr/bin/env python3
"""Accuracy by output position (the blind spot made visible). At Qwen2.5-32B and N = 32, the
index-scored accuracy at each output position, averaged over trials and alphabet or modulus values,
for the no-marker run and for markers every 4 and every 8 characters: (a) a position-local digit
task, where every position gains; (b) a one-step dependency task, where the position right after
each marker (4, 8, 12, ... or 8, 16, 24) drops while the others gain.

Output: figs/fig_position_curve.pdf
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, INK, GRAY  # noqa: E402
from analyze_per_position import load  # noqa: E402
from analyze_aligned_scoring import index_vector, clean  # noqa: E402

MODEL = ("Qwen2.5 32B", "Qwen2.5-32B-Instruct")
N = 32
PANELS = [("random_permutation_digits", "(a) Position-local task (random permutation, digits)"),
          ("running_xor", "(b) One-step dependency task (running XOR)")]
LINES = [(0, "No markers", INK, "--"), (4, "Markers every 4", "#D55E00", "-"), (8, "Markers every 8", "#0072B2", "-")]


def curve(d, k):
    x = d[(d.Seq_Length.astype(int) == N) & (d.Interval.astype(int) == k) & (d.Method == ("Base" if k == 0 else "Cellular_Bounds"))]
    vs = np.array([index_vector(clean(r.Ground_Truth), clean(r.LLM_Output))[:N] for _, r in x.iterrows() if len(clean(r.Ground_Truth)) == N])
    return vs.mean(0), 1.96 * vs.std(0, ddof=1) / np.sqrt(len(vs)), len(vs)


def main():
    set_rc()
    fig, axes = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH, 2.3), sharey=True)
    for ax, (task, title) in zip(axes, PANELS):
        d = load(task, MODEL[1])
        pos = np.arange(1, N + 1)
        for k, lab, col, ls in LINES:
            m, ci, n = curve(d, k)
            ax.plot(pos, m, color=col, linestyle=ls, linewidth=1.1, label=f"{lab} (n = {n})", zorder=3)
            ax.fill_between(pos, m - ci, m + ci, color=col, alpha=0.15, linewidth=0, zorder=2)
            if k:
                post = [i for i in pos if (i - 1) % k == 0 and i > 1]
                ax.plot(post, m[[i - 1 for i in post]], linestyle="none", marker="o", markersize=3.2, color=col, zorder=4)
        ax.set_title(title)
        ax.set_xlabel("Output position")
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
        ax.set_xlim(0.5, N + 0.5); ax.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Character accuracy")
    axes[1].legend(loc="lower left", fontsize=6.5)
    axes[1].text(0.98, 0.97, "dots: position right after a marker", transform=axes[1].transAxes, ha="right", va="top", fontsize=6, color=GRAY)
    fig.tight_layout()
    savefig(fig, REPO / "figs" / "fig_position_curve.pdf")


if __name__ == "__main__":
    main()
