#!/usr/bin/env python3
"""Figure for the "External benchmark tasks" section: clrs_minimum and
clrs_kadane (CLRS-Text, Markeeva et al. arXiv:2406.04229) run through
Gemini (gemini-3.5-flash), 3 trials per condition, N up to 128. Same
by-density-line convention as make_fig_external_tier1_checks.py (Base
dashed black, each tested k its own colored line) rather than a "mean
over k" pooled line, for the same reason established there: pooling
would dilute exactly the density effect the accompanying table
highlights.

Neither task has a fixed-cardinality random-chance line: their answer
space (a running-min index up to N, or a (start,end) tuple pair) grows
with N and position, so "1/alphabet size" isn't well-defined the way it
is for every char-level task elsewhere in this report -- no chance line
is drawn, rather than drawing a misleading placeholder.

Originally a 3rd panel (parity, modsum at modulus=2), removed along with
the rest of the parity write-up: its Base-vs-Bounds direction flipped
across models (positive at 7B/Gemini, significantly negative at 32B)
with no identified explanation, and repeated attempts to explain it
clearly to the report's own author didn't land -- an unexplained result
that can't be clearly communicated isn't earning its place in the
report, independent of whether it's "real."
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, set_power_of_2_xticks

set_rc()

# five distinct steps (k = 4 and k = 8 previously shared one color)
SEQ_BLUE_ORDINAL = ["#b9d7ee", "#74afdc", "#2f88c8", "#005a96", "#002a4a"]


def plot_by_density(ax, df, chance, title):
    base = df[df.Method == "Base"].groupby("Seq_Length")["TAR"].mean()
    if chance is not None:
        ax.axhline(chance, color="#999999", linestyle=":", linewidth=1, alpha=0.85, label="Random chance")
    ax.plot(base.index, base.values, color="black", linestyle="--", marker="o", markersize=4, linewidth=1.3, label="Base")
    ks = sorted(df[df.Method != "Base"].Interval.unique())
    for color, k in zip(SEQ_BLUE_ORDINAL, ks):
        sub = df[df.Interval == k].groupby("Seq_Length")["TAR"].mean()
        ax.plot(sub.index, sub.values, color=color, marker="s", markersize=3.5, linewidth=1.3, label=f"k={int(k)}")
    set_power_of_2_xticks(ax, sorted(df.Seq_Length.unique()), fontsize=6.5)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("TAR")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=8.5)


dfs = [pd.read_csv(REPO_ROOT / "results_clean" / "gemini_pilot" / f"gemini-3.5-flash_full_grid_trial{t}.csv") for t in [0, 1, 2]]
full = pd.concat(dfs, ignore_index=True)
full = full[full.Interval != 3]

# clrs_kadane panel removed: it is at chance (the text keeps a one-sentence disclosure).
fig, ax = plt.subplots(1, 1, figsize=(MULTI_COL_WIDTH * 0.4, 2.8))

plot_by_density(ax, full[full.Task == "clrs_minimum"], chance=None, title="clrs_minimum")

ax.legend(loc="lower left", frameon=False, fontsize=5.5, ncol=1)
fig.suptitle("CLRS-Text tasks on Gemini (gemini-3.5-flash), 3 trials/condition, by boundary density k", fontsize=9, y=1.05)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "external_benchmark_gemini.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("Wrote external_benchmark_gemini")
