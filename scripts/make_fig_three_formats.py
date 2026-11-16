#!/usr/bin/env python3
"""Figures for the tokenization analysis: merged letters vs spaced letters vs digits.

fig_three_formats: per model, token accuracy without and with markers for each format,
  mean over the three position-local tasks (N <= 64, k in {1,2,4,8,16}); the no-marker value is
  length-matched to the marker runs (tasks/common/pooled.py).
fig_profile: at the four strong models (Qwen2.5-14B/32B, Qwen3.8-27B, Gemma-4-31B), the marker
  effect (Bounds - Base) by sequence length, and by marker spacing at N = 64.

Reads the raw grids; also writes results_clean/_summary/three_formats_by_N_k.csv.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.pooled import pooled_delta  # noqa: E402
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH  # noqa: E402

MODELS = {"Qwen2.5 7B": "Qwen2.5-7B-Instruct", "Qwen2.5 14B": "Qwen2.5-14B-Instruct",
          "Qwen2.5 32B": "Qwen2.5-32B-Instruct", "Mistral 7B": "Mistral-7B-Instruct-v0.3",
          "Phi-3.5 mini": "Phi-3.5-mini-instruct", "Qwen3.8 27B": "Qwen3.8-27B", "Gemma 4 31B": "gemma-4-31B-it",
          "Gemini 3.6 Flash": "gemini36"}
GEMINI36 = REPO / "results_clean" / "gemini_pilot" / "gemini-3.6-flash_POOLED.csv"
STRONG = ["Qwen2.5 14B", "Qwen2.5 32B", "Qwen3.8 27B", "Gemma 4 31B"]
TASKS = ["cyclic_shift", "random_permutation", "conditional_shift"]
FORMATS = [("merged", "", "Letters", "#0072B2"), ("spaced", "_spaced", "Spaced letters", "#D55E00"),
           ("digits", "_digits", "Digits", "#009E73")]
MARKERS = {"merged": "o", "spaced": "s", "digits": "^"}  # shape as well as color
GRAY, INK = "#6e6e6e", "#1a1a1a"


def load(task, tag):
    if tag == "gemini36":
        if not GEMINI36.exists():
            return None
        d = pd.read_csv(GEMINI36); d = d[d.Task == task].rename(columns={"Axis_Val": "Alphabet_Size"})
        return d[(d.Interval != 3) & (d.Seq_Length <= 64)] if len(d) else None
    # the base model's own files only ({tag}.csv, {tag}_*.csv): the LoRA runs saved next to them
    # ({tag}-lorabase*.csv, {tag}-lorasep.csv) are other models and must not be mixed in
    fs = [f for f in sorted(glob.glob(str(REPO / "results_clean" / task / "grid" / f"{tag}*.csv")))
          if Path(f).name == f"{tag}.csv" or Path(f).name.startswith(f"{tag}_")
          if "POOLED" not in f and "N128" not in f]
    x = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(
        ["Method", "Interval", "Seq_Length", "Trial", "Alphabet_Size"])
    return x[(x.Interval != 3) & (x.Seq_Length <= 64)]


def collect():
    pooled, cells = [], []
    for m, tag in MODELS.items():
        for t in TASKS:
            for fmt, suf, _, _ in FORMATS:
                x = load(t + suf, tag)
                if x is None:
                    continue
                r = pooled_delta(x)  # length-matched (tasks/common/pooled.py)
                c = x[x.Method == "Cellular_Bounds"].TAR
                pooled.append(dict(model=m, task=t, fmt=fmt, base=r["base"], bounds=r["bounds"],
                                   bounds_se=c.sem()))
                for N, g in x.groupby("Seq_Length"):
                    bb = g[g.Method == "Base"].TAR.mean()
                    for k, h in g[g.Method == "Cellular_Bounds"].groupby("Interval"):
                        cells.append(dict(model=m, task=t, fmt=fmt, N=N, k=k, base=bb, bounds=h.TAR.mean()))
    cells = pd.DataFrame(cells)
    cells["delta"] = cells.bounds - cells.base
    cells.to_csv(REPO / "results_clean" / "_summary" / "three_formats_by_N_k.csv", index=False)
    return pd.DataFrame(pooled), cells


MODEL_COLORS = {"Qwen2.5 7B": "#9ecae1", "Qwen2.5 14B": "#4292c6", "Qwen2.5 32B": "#08306b", "Mistral 7B": "#E69F00",
                "Phi-3.5 mini": "#CC79A7", "Qwen3.8 27B": "#009E73", "Gemma 4 31B": "#D55E00", "Gemini 3.6 Flash": "#6e6e6e"}
SHORT = {"Qwen2.5 7B": "Qwen2.5 7B", "Qwen2.5 14B": "Qwen2.5 14B", "Qwen2.5 32B": "Qwen2.5 32B", "Mistral 7B": "Mistral 7B",
         "Phi-3.5 mini": "Phi-3.5 mini", "Qwen3.8 27B": "Qwen3.8 27B", "Gemma 4 31B": "Gemma 4 31B", "Gemini 3.6 Flash": "Gemini 3.6"}


def fig_three_formats(p):
    """Slopegraph, one panel per format: for each model a line from its no-marker accuracy to its
    accuracy with markers (95% interval at the marker end), mean over the three position-local tasks."""
    fig, axes = plt.subplots(1, 3, figsize=(MULTI_COL_WIDTH, 3.0), sharey=True)
    for letter, ax, (fmt, _, lab, _) in zip("abc", axes, FORMATS):
        rows = []
        for m in MODELS:
            s = p[(p.model == m) & (p.fmt == fmt)]
            if s.empty:
                continue
            b, c = s.base.mean(), s.bounds.mean()
            ci = 1.96 * np.sqrt((s.bounds_se ** 2).sum()) / len(s)
            col = MODEL_COLORS[m]
            ax.plot([0, 1], [b, c], color=col, linewidth=1.2, zorder=3)
            ax.plot(0, b, marker="o", markersize=3.2, color=col, zorder=4)
            ax.errorbar(1, c, yerr=ci, fmt="o", markersize=3.2, color=col, ecolor=col, elinewidth=0.7, capsize=1.5, zorder=4)
            rows.append((c, m, col))
        # right-hand labels, spread apart (in both directions, kept under the top of the axis)
        # where they would collide
        rows.sort()
        ys = [r[0] for r in rows]
        # the gap must exceed the label's own line height or neighbours still touch:
        # 0.048 of the 0.77 y-range is about 8.5 pt at this panel height, against a 7 pt font
        gap, top = 0.048, 1.0
        for _ in range(500):  # symmetric spreading of colliding neighbours
            moved = False
            for k in range(1, len(ys)):
                d = ys[k] - ys[k - 1]
                if d < gap - 1e-9:
                    ys[k] += (gap - d) / 2; ys[k - 1] -= (gap - d) / 2; moved = True
            if not moved:
                break
        if ys[-1] > top:      # keep the top label inside the axis, pushing the ones below down
            ys[-1] = top
            for k in range(len(ys) - 2, -1, -1):
                ys[k] = min(ys[k], ys[k + 1] - gap)
        for (c, m, col), y in zip(rows, ys):
            ax.annotate(SHORT[m], xy=(1, c), xytext=(1.08, y), fontsize=7, color=col, va="center",
                        arrowprops=dict(arrowstyle="-", color=col, linewidth=0.4, alpha=0.6, shrinkA=0, shrinkB=1))
        ax.set_title(f"({letter}) {lab}")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No markers", "Markers"])
        ax.set_xlim(-0.15, 1.75); ax.set_ylim(0.25, 1.02)
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.5)
    axes[0].set_ylabel("Character accuracy")
    fig.tight_layout()
    savefig(fig, REPO / "figs" / "fig_three_formats.pdf")
    fig.savefig(REPO / "figs" / "png" / "fig_three_formats.png", dpi=200, bbox_inches="tight")


def fig_profile(cells):
    s = cells[cells.model.isin(STRONG)]
    fig, axes = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH, 2.4), sharey=True)
    for fmt, _, lab, col in FORMATS:
        x = s[s.fmt == fmt]
        # one value per (model, task) per N: mean over k; then mean and 95% CI over the 12 pairs
        byN = x.groupby(["model", "task", "N"]).delta.mean().groupby("N").agg(["mean", "sem"])
        axes[0].errorbar(byN.index, byN["mean"], yerr=1.96 * byN["sem"], color=col, marker=MARKERS[fmt],
                         markersize=3.5, linewidth=1.6, capsize=1.5, label=lab)
        byk = x[x.N == 64].groupby(["model", "task", "k"]).delta.mean().groupby("k").agg(["mean", "sem"])
        axes[1].errorbar(byk.index, byk["mean"], yerr=1.96 * byk["sem"], color=col, marker=MARKERS[fmt],
                         markersize=3.5, linewidth=1.6, capsize=1.5, label=lab)
    for ax, xl in [(axes[0], "Sequence length $N$"), (axes[1], "Marker spacing $k$")]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xl)
        ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_xticks([4, 8, 16, 32, 64]); axes[0].set_xticklabels(["4", "8", "16", "32", "64"])
    axes[1].set_xticks([1, 2, 4, 8, 16]); axes[1].set_xticklabels(["1", "2", "4", "8", "16"])
    axes[0].set_ylabel("Gain from markers")
    axes[0].set_title("(a) By sequence length")
    axes[1].set_title("(b) By marker spacing, $N = 64$")
    fig.tight_layout()
    # one shared legend, outside the axes, below the panels (the house placement)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    savefig(fig, REPO / "figs" / "fig_profile.pdf")
    fig.savefig(REPO / "figs" / "png" / "fig_profile.png", dpi=200, bbox_inches="tight")


def fig_profile_by_task(cells):
    """Appendix: the length profile per task (mean over the four strong models)."""
    s = cells[cells.model.isin(STRONG)]
    fig, axes = plt.subplots(1, 3, figsize=(MULTI_COL_WIDTH, 2.3), sharey=True)
    for ax, t in zip(axes, TASKS):
        for fmt, _, lab, col in FORMATS:
            x = s[(s.fmt == fmt) & (s.task == t)]
            byN = x.groupby(["model", "N"]).delta.mean().groupby("N").agg(["mean", "sem"])
            ax.errorbar(byN.index, byN["mean"], yerr=1.96 * byN["sem"], color=col, marker=MARKERS[fmt],
                        markersize=3.2, linewidth=1.5, capsize=1.5, label=lab)
        ax.set_title(t, fontsize=7.5, family="monospace")
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
        ax.set_xlabel("Sequence length $N$")
        ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Gain from markers\n(Bounds $-$ Base)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout()
    savefig(fig, REPO / "figs" / "fig_profile_by_task.pdf")
    fig.savefig(REPO / "figs" / "png" / "fig_profile_by_task.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    set_rc()
    pooled, cells = collect()
    fig_three_formats(pooled)
    fig_profile(cells)
    fig_profile_by_task(cells)
    print("Wrote figs/fig_three_formats.pdf and figs/fig_profile.pdf")
