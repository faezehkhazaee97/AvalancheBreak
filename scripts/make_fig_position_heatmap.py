#!/usr/bin/env python3
"""Accuracy by output position: (a) curves at one model, (b, c) heatmaps across models.

(a) Qwen2.5-32B, N = 32: index-scored accuracy at each output position without markers and with
    markers every 4 and every 8 characters, on a position-local digit task and on running_xor.
(b, c) Every model, N = 32, markers every 4: the change in accuracy at each output position
    relative to the no-marker run at the same position, pooled over the four dependency tasks (b)
    and the three position-local digit tasks (c). Columns 5, 9, 13, ... follow a marker.

Output: figs/fig_position_curve.pdf
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, INK, GRAY  # noqa: E402
from analyze_per_position import load, MODELS  # noqa: E402
from analyze_aligned_scoring import index_vector, clean  # noqa: E402

N, K = 32, 4
DEP = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]
LOC = ["cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"]
CMAP = LinearSegmentedColormap.from_list("hurts_helps", ["#D55E00", "#f7f7f7", "#0072B2"])
SHORT = {"Qwen2.5 7B": "Qwen2.5 7B", "Qwen2.5 14B": "Qwen2.5 14B", "Qwen2.5 32B": "Qwen2.5 32B", "Mistral 7B": "Mistral 7B",
         "Phi-3.5 mini": "Phi-3.5 mini", "Qwen3.8 27B": "Qwen3.8 27B", "Gemma 4 31B": "Gemma 4 31B",
         "Gemini 3.5 Flash": "Gemini 3.5", "Gemini 3.6 Flash": "Gemini 3.6"}


def vectors(d, k):
    x = d[(d.Seq_Length.astype(int) == N) & (d.Interval.astype(int) == k) & (d.Method == ("Base" if k == 0 else "Cellular_Bounds"))]
    vs = [index_vector(clean(r.Ground_Truth), clean(r.LLM_Output))[:N] for _, r in x.iterrows() if len(clean(r.Ground_Truth)) == N]
    return np.array(vs) if vs else None


def main():
    set_rc()
    fig = plt.figure(figsize=(MULTI_COL_WIDTH, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.55, wspace=0.12)
    # (a) curves at Qwen2.5-32B
    lines = [(0, "No markers", INK, "--"), (4, "Markers every 4", "#D55E00", "-"), (8, "Markers every 8", "#0072B2", "-")]
    for col, (task, title) in enumerate([("random_permutation_digits", "(a) Position-local task, Qwen2.5-32B"), ("running_xor", "(b) One-step dependency task, Qwen2.5-32B")]):
        ax = fig.add_subplot(gs[0, col]); d = load(task, "Qwen2.5-32B-Instruct"); pos = np.arange(1, N + 1)
        for k, lab, c, ls in lines:
            vs = vectors(d, k); m = vs.mean(0); ci = 1.96 * vs.std(0, ddof=1) / np.sqrt(len(vs))
            ax.plot(pos, m, color=c, linestyle=ls, linewidth=1.1, label=lab, zorder=3)
            ax.fill_between(pos, m - ci, m + ci, color=c, alpha=0.15, linewidth=0, zorder=2)
            if k:
                post = [i for i in pos if (i - 1) % k == 0 and i > 1]
                ax.plot(post, m[[i - 1 for i in post]], linestyle="none", marker="o", markersize=3, color=c, zorder=4)
        ax.set_title(title); ax.set_xlabel("Output position"); ax.set_xlim(0.5, N + 0.5); ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 8, 16, 24, 32])
        if col == 0:
            ax.set_ylabel("Character accuracy")
        else:
            ax.set_yticklabels([])
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    # (c, d) heatmaps across models
    models = [(lab, tag) for lab, tag in MODELS]
    for col, (tasks, title) in enumerate([(LOC, "(c) Position-local digit tasks, every model"), (DEP, "(d) Dependency tasks, every model")]):
        ax = fig.add_subplot(gs[1, col]); grid = np.full((len(models), N), np.nan)
        for i, (lab, tag) in enumerate(models):
            vals = []
            for t in tasks:
                d = load(t, tag)
                if d is None:
                    continue
                b, m = vectors(d, 0), vectors(d, K)
                if b is None or m is None:
                    continue
                vals.append(m.mean(0) - b.mean(0))
            if vals:
                grid[i] = np.mean(vals, axis=0)
        im = ax.imshow(np.ma.masked_invalid(grid), cmap=CMAP, vmin=-0.5, vmax=0.5, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(models))); ax.set_yticklabels([SHORT[l] for l, _ in models] if col == 0 else [""] * len(models), fontsize=6.5)
        ax.set_xticks([0, 7, 15, 23, 31]); ax.set_xticklabels([1, 8, 16, 24, 32]); ax.set_xlabel("Output position")
        for p in range(K, N, K):
            ax.axvline(p - 0.5, color="white", linewidth=0.8)
        ax.set_title(title); ax.grid(False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
    cax = fig.add_axes([0.92, 0.11, 0.012, 0.3])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Change in accuracy with markers every 4", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.13, right=0.9, top=0.9, bottom=0.08)
    savefig(fig, REPO / "figs" / "fig_position_curve.pdf")


if __name__ == "__main__":
    main()
