#!/usr/bin/env python3
"""Accuracy over the length-by-spacing grid, one small heatmap per model (the form used for length
generalization: McLeish et al. 2024, Fig. 1). Rows: marker spacing k (0 = no markers, then 1, 2, 4,
8, 16); columns: sequence length N (4 to 64); color: character accuracy, mean over the block's tasks
and alphabet or modulus values. Top row of panels: the three position-local digit tasks, where the
marker rows brighten relative to the no-marker row as N grows. Bottom row: the four dependency
tasks, where the rows with k >= 2 darken relative to k = 0 and k = 1 at every length.

Output: figs/fig_length_spacing_grid.pdf
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, INK  # noqa: E402
from analyze_per_position import load  # noqa: E402

MODELS = [("Qwen2.5 7B", "Qwen2.5-7B-Instruct"), ("Qwen2.5 14B", "Qwen2.5-14B-Instruct"), ("Qwen2.5 32B", "Qwen2.5-32B-Instruct"),
          ("Mistral 7B", "Mistral-7B-Instruct-v0.3"), ("Phi-3.5 mini", "Phi-3.5-mini-instruct"), ("Qwen3.8 27B", "Qwen3.8-27B"),
          ("Gemma 4 31B", "gemma-4-31B-it"), ("Gemini 3.6", "gemini36")]
BLOCKS = [("Position-local digit tasks", ["cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"]),
          ("Dependency tasks", ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"])]
NS, KS = [4, 8, 16, 32, 64], [0, 1, 2, 4, 8, 16]
CMAP = LinearSegmentedColormap.from_list("acc", ["#fff5eb", "#fdd0a2", "#9ecae1", "#2171b5", "#08306b"])


def grid(tag, tasks):
    g = np.full((len(KS), len(NS)), np.nan); cnt = np.zeros_like(g)
    for t in tasks:
        d = load(t, tag)
        if d is None:
            continue
        d = d.assign(N=d.Seq_Length.astype(int), k=d.Interval.astype(int), TAR=d.TAR.astype(float))
        for (N, k), x in d.groupby(["N", "k"]):
            if N in NS and k in KS:
                i, j = KS.index(k), NS.index(N)
                g[i, j] = (0 if np.isnan(g[i, j]) else g[i, j]) + x.TAR.mean(); cnt[i, j] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, g / np.maximum(cnt, 1), np.nan)


def main():
    set_rc()
    from matplotlib.transforms import offset_copy
    fig, axes = plt.subplots(len(BLOCKS), len(MODELS), figsize=(MULTI_COL_WIDTH, 3.1), sharex=True, sharey=True)
    for r, (bname, tasks) in enumerate(BLOCKS):
        for c, (label, tag) in enumerate(MODELS):
            ax = axes[r][c]
            g = grid(tag, tasks)
            im = ax.imshow(np.ma.masked_invalid(g), cmap=CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            ax.axhline(0.5, color="white", linewidth=1.2)          # separates the no-marker row
            ax.set_xticks(range(len(NS))); ax.set_xticklabels([str(n) if n in (4, 16, 64) else "" for n in NS])   # every other label: room at 7 pt
            ax.set_yticks(range(len(KS))); ax.set_yticklabels(["none"] + [str(k) for k in KS[1:]])
            ax.tick_params(axis="x", pad=1.5)
            ax.grid(False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(True)
            if r == 0:
                ax.set_title(label, loc="center", fontsize=7)    # column header: the model
        axes[r][0].set_ylabel("Marker spacing $k$")
    cax = fig.add_axes([0.915, 0.14, 0.012, 0.68]); cb = fig.colorbar(im, cax=cax)
    cb.set_label("Character accuracy")
    fig.subplots_adjust(left=0.08, right=0.9, top=0.84, bottom=0.13, wspace=0.12, hspace=0.42)
    fig.supxlabel("Sequence length $N$", y=0.02, fontsize=plt.rcParams["axes.labelsize"])
    # Block titles, one per row of panels, each starting with its panel letter in Figure 1's style
    # (8.5 pt, normal weight), left-aligned with the row's first panel: (a) above the model names,
    # (b) at the usual title position of its row.
    pad = plt.rcParams["axes.titlepad"]
    for r, (letter, (bname, _)) in enumerate(zip("ab", BLOCKS)):
        lift = pad + (7 * 1.2 + 3 if r == 0 else 0)
        tr = offset_copy(axes[r][0].transAxes, fig=fig, x=0, y=lift, units="points")
        fig.text(0.0, 1.0, f"({letter}) {bname}", transform=tr, ha="left", va="baseline",
                 fontsize=plt.rcParams["axes.titlesize"])
    savefig(fig, REPO / "figs" / "fig_length_spacing_grid.pdf")


if __name__ == "__main__":
    main()
