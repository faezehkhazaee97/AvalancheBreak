#!/usr/bin/env python3
"""Figure 1 of the paper: the three effects of content-free markers at a glance.

(a) Re-tokenization: real Qwen2.5 tokenization of one letter string, merged, with a marker
    every 4 letters, and spaced (one token per letter).
(b) Tokenization-independent gain: the marker effect (Bounds - Base) by sequence length on
    spaced letters and digits, mean over the four strong models and three position-local
    tasks (from results_clean/_summary/three_formats_by_N_k.csv, written by
    scripts/make_fig_three_formats.py); same data as Figure 3a.
(c) Blind spot: running_xor dependencies; the one that crosses a marker fails.

Output: figs/fig0_overview.pdf
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH  # noqa: E402

INK, GRAY, LIGHT = "#0b0b0b", "#78776f", "#ecebe6"
MERGED_FILL = "#d7e6f7"          # multi-letter token
MARK = "#2a78d6"                 # marker
BAD = "#c8402f"                  # failing dependency
FORMATS = [("spaced", "Spaced letters", "#eb6834"), ("digits", "Digits", "#1baf7a")]
MARKERS = {"spaced": "s", "digits": "^"}
STRONG = ["Qwen2.5 14B", "Qwen2.5 32B", "Qwen3.8 27B", "Gemma 4 31B"]

# Qwen2.5 tokenizer output for these strings (checked with AutoTokenizer, Qwen2.5-32B-Instruct)
TOKENS = [
    ("No markers", ["C", "AB", "DD", "AC", "BB", "ADC"]),
    ("Markers, $k{=}4$", ["C", "AB", "D", "|", "DAC", "B", "|", "B", "ADC"]),
    ("Spaced", list("CABDDACBBADC")),
]


def panel_tokens(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(a) Letters: markers re-tokenize", loc="left", fontsize=8.5)
    unit, x0 = 0.5, 3.0
    for row, (label, toks) in enumerate(TOKENS):
        y = 7.6 - row * 2.55
        ax.text(x0 - 0.25, y, label, ha="right", va="center", fontsize=7, color=INK)
        x = x0
        for t in toks:
            w = unit * len(t) * (0.75 if t == "|" else 1)
            fill = LIGHT if t == "|" else (MERGED_FILL if len(t) > 1 else "white")
            ax.add_patch(FancyBboxPatch((x + 0.04, y - 0.62), w - 0.08, 1.24,
                                        boxstyle="round,pad=0,rounding_size=0.12",
                                        facecolor=fill, edgecolor=MARK if t == "|" else GRAY, linewidth=0.6))
            ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=6.6,
                    color=MARK if t == "|" else INK, family="monospace", fontweight="bold" if t == "|" else None)
            x += w
    ax.add_patch(FancyBboxPatch((x0, 0.15), 0.5, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=MERGED_FILL, edgecolor=GRAY, linewidth=0.6))
    ax.text(x0 + 0.7, 0.4, "multi-letter token", va="center", fontsize=6.3, color=GRAY)


def panel_profile(ax):
    cells = pd.read_csv(REPO / "results_clean" / "_summary" / "three_formats_by_N_k.csv")
    s = cells[cells.model.isin(STRONG)]
    for fmt, lab, col in FORMATS:
        byN = s[s.fmt == fmt].groupby(["model", "task", "N"]).delta.mean().groupby("N").agg(["mean", "sem"])
        ax.errorbar(byN.index, byN["mean"], yerr=1.96 * byN["sem"], color=col, marker=MARKERS[fmt], markersize=3.4,
                    linewidth=1.6, capsize=1.5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Gain from markers")
    ax.set_title("(b) Gain on long sequences", loc="left", fontsize=8.5)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)


def panel_blindspot(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(c) Dependency tasks: blind spot", loc="left", fontsize=8.5)
    x_in = list("1011") + ["|"] + list("0110")
    y_out = ["1", "1", "1", "0", "|", "1", "1", "0", "1"]   # XOR with the previous input
    xs = [2.4 + i * 0.85 for i in range(len(x_in))]
    yi, yo = 7.4, 2.9
    ax.text(0.0, yi, "in", ha="left", va="center", fontsize=6.5, color=GRAY)
    ax.text(0.0, yo, "out", ha="left", va="center", fontsize=6.5, color=GRAY)
    for x, a, b in zip(xs, x_in, y_out):
        c = MARK if a == "|" else INK
        ax.text(x, yi, a, ha="center", va="center", fontsize=8, family="monospace", color=c,
                fontweight="bold" if a == "|" else None)
        ax.text(x, yo, b, ha="center", va="center", fontsize=8, family="monospace", color=c,
                fontweight="bold" if a == "|" else None)
    digits = [i for i, a in enumerate(x_in) if a != "|"]
    for j in range(1, len(digits)):
        src, dst = digits[j - 1], digits[j]
        crosses = dst - src > 1
        ax.add_patch(FancyArrowPatch((xs[src] + 0.12, yi - 0.75), (xs[dst] - 0.05, yo + 0.75),
                                     arrowstyle="-|>", mutation_scale=6, linewidth=1.3 if crosses else 0.7,
                                     color=BAD if crosses else GRAY, linestyle="--" if crosses else "-"))
    post = digits[4]
    ax.text(xs[post] + 0.3, yo - 1.25, "post-marker:\nnear chance", ha="right", va="top", fontsize=6.3, color=BAD)
    ax.text(xs[post + 1] - 0.2, yo - 1.25, "next:\nrecovers", ha="left", va="top", fontsize=6.3, color=GRAY)


def main():
    set_rc()
    fig = plt.figure(figsize=(MULTI_COL_WIDTH, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.8, 1.0], wspace=0.42)
    panel_tokens(fig.add_subplot(gs[0]))
    panel_profile(fig.add_subplot(gs[1]))
    panel_blindspot(fig.add_subplot(gs[2]))
    savefig(fig, REPO / "figs" / "fig0_overview.pdf")
    fig.savefig(REPO / "figs" / "png" / "fig0_overview.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
