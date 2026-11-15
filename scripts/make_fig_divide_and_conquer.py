#!/usr/bin/env python3
"""Figure 7: Divide-and-conquer comparison (figs/divide_and_conquer.pdf).
TAR vs. N and wall-clock vs. N for Base / Cellular_Bounds / true
divide-and-conquer, side by side. Qwen2.5-32B (rebuilt from the original
Qwen2.5-7B version to match the main paper's single-model story), now
extended to N=128. Replaces old Figure 19.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, set_power_of_2_xticks

set_rc()

SRC = REPO_ROOT / "results" / "Qwen2.5-32B-Instruct" / "divide_conquer_grid.csv"
print(f"Provenance -- canonical file used for Figure 7: {SRC.relative_to(REPO_ROOT)}")
df = pd.read_csv(SRC)

COLORS = {"Base": "black", "Cellular_Bounds": "#D55E00", "Divide_Conquer": "#009E73"}  # Okabe-Ito
LABELS = {"Base": "Base", "Cellular_Bounds": "Cellular_Bounds", "Divide_Conquer": "Divide-and-conquer"}
STYLES = {"Base": "--", "Cellular_Bounds": "-", "Divide_Conquer": "-"}
MARKERS = {"Base": "o", "Cellular_Bounds": "s", "Divide_Conquer": "^"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH, 3.0))

for cond in ["Base", "Cellular_Bounds", "Divide_Conquer"]:
    sub = df[df.Condition == cond]
    tar = sub.groupby("Seq_Length")["TAR"].agg(["mean", "sem"])
    wt = sub.groupby("Seq_Length")["Wall_Time_Sec"].agg(["mean", "sem"])

    ax1.plot(tar.index, tar["mean"], color=COLORS[cond], linestyle=STYLES[cond],
              marker=MARKERS[cond], markersize=4, linewidth=1.5, label=LABELS[cond])
    ax1.fill_between(tar.index, tar["mean"] - tar["sem"], tar["mean"] + tar["sem"],
                       color=COLORS[cond], alpha=0.15)

    ax2.plot(wt.index, wt["mean"], color=COLORS[cond], linestyle=STYLES[cond],
              marker=MARKERS[cond], markersize=4, linewidth=1.5, label=LABELS[cond])
    ax2.fill_between(wt.index, wt["mean"] - wt["sem"], wt["mean"] + wt["sem"],
                       color=COLORS[cond], alpha=0.15)

ax1.set_xlabel("Sequence length N")
ax1.set_ylabel("TAR")
ax1.set_ylim(0, 1.05)
set_power_of_2_xticks(ax1, df.Seq_Length.unique(), fontsize=6.5)
ax1.legend(loc="lower left", frameon=False, fontsize=6.5)
ax1.set_title("Accuracy")

ax2.set_xlabel("Sequence length N")
ax2.set_ylabel("Wall-clock time (s)")
set_power_of_2_xticks(ax2, df.Seq_Length.unique(), fontsize=6.5)
ax2.set_yscale("log")
ax2.set_yticks([1, 2, 5, 10, 20, 50])
ax2.set_yticklabels(["1", "2", "5", "10", "20", "50"])
ax2.minorticks_off()
ax2.legend(loc="upper left", frameon=False, fontsize=6.5)
ax2.set_title("Cost")

fig.suptitle("Qwen2.5-32B: Base vs. Cellular_Bounds vs. true divide-and-conquer", y=1.03)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "divide_and_conquer.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
