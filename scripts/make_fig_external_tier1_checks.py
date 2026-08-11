#!/usr/bin/env python3
"""Figures for the "External Tier-1 checks" section: vowel/consonant
tagging (7B + 32B), ProcBench-Substitute (7B), string copying (7B).
Simple Base-vs-Cellular_Bounds line plots, matching the visual
convention used for every other task card in this report -- these three
previously had tables only, no figures.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, set_power_of_2_xticks

set_rc()

CAT_BLUE = "#2a78d6"
CAT_ORANGE = "#eb6834"

# ---- vowel/consonant tagging: 7B vs 32B, one panel each ----
fig, axes = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH * 0.65, 2.8))
for ax, model, color in [(axes[0], "Qwen2.5-7B-Instruct", CAT_BLUE), (axes[1], "Qwen2.5-32B-Instruct", CAT_ORANGE)]:
    df = pd.read_csv(REPO_ROOT / "results" / model / "vowel_tagging_boundary_sweep.csv")
    base = df[df.Interval == 0].groupby("Seq_Length")["TAR"].mean()
    bnd = df[df.Interval != 0].groupby("Seq_Length")["TAR"].mean()
    ax.plot(base.index, base.values, color="black", linestyle="--", marker="o", markersize=4, linewidth=1.3, label="Base")
    ax.plot(bnd.index, bnd.values, color=color, linestyle="-", marker="s", markersize=4, linewidth=1.6, label="Cellular_Bounds\n(mean over k)")
    set_power_of_2_xticks(ax, base.index, fontsize=6.5)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("TAR")
    ax.set_ylim(0, 1.0)
    ax.set_title(model.replace("-Instruct", ""), fontsize=8.5)
    ax.legend(loc="upper right", frameon=False, fontsize=6)
fig.suptitle("vowel/consonant tagging: Base vs. Cellular_Bounds, by model", fontsize=9, y=1.03)
fig.tight_layout()
OUT = REPO_ROOT / "figs" / "external_vowel_tagging.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- ProcBench-Substitute: 7B only ----
fig, ax = plt.subplots(figsize=(MULTI_COL_WIDTH * 0.42, 2.8))
df = pd.read_csv(REPO_ROOT / "results" / "external_tasks" / "procbench_substitute" / "Qwen2.5-7B-Instruct_grid.csv")
df = df[df.Interval != 3]
base = df[df.Method == "Base"].groupby("Seq_Length")["TAR"].mean()
bnd = df[df.Method == "Cellular_Bounds"].groupby("Seq_Length")["TAR"].mean()
ax.plot(base.index, base.values, color="black", linestyle="--", marker="o", markersize=4, linewidth=1.3, label="Base")
ax.plot(bnd.index, bnd.values, color=CAT_BLUE, linestyle="-", marker="s", markersize=4, linewidth=1.6, label="Cellular_Bounds\n(mean over k)")
set_power_of_2_xticks(ax, base.index, fontsize=6.5)
ax.set_xlabel("Sequence length N")
ax.set_ylabel("TAR")
ax.set_ylim(0, 0.75)
ax.set_title("ProcBench-Substitute (Qwen2.5-7B)", fontsize=8.5)
ax.legend(loc="upper right", frameon=False, fontsize=6.5)
fig.tight_layout()
OUT = REPO_ROOT / "figs" / "external_procbench.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- String copying: 7B only ----
fig, ax = plt.subplots(figsize=(MULTI_COL_WIDTH * 0.42, 2.8))
df = pd.read_csv(REPO_ROOT / "results" / "external_tasks" / "string_copy" / "Qwen2.5-7B-Instruct_grid.csv")
df = df[df.Interval != 3]
base = df[df.Method == "Base"].groupby("Seq_Length")["TAR"].mean()
bnd = df[df.Method == "Cellular_Bounds"].groupby("Seq_Length")["TAR"].mean()
ax.plot(base.index, base.values, color="black", linestyle="--", marker="o", markersize=4, linewidth=1.3, label="Base")
ax.plot(bnd.index, bnd.values, color=CAT_ORANGE, linestyle="-", marker="s", markersize=4, linewidth=1.6, label="Cellular_Bounds\n(mean over k)")
set_power_of_2_xticks(ax, base.index, fontsize=6.5)
ax.set_xlabel("Sequence length N")
ax.set_ylabel("TAR")
ax.set_ylim(0.85, 1.02)
ax.set_title("String copying (Qwen2.5-7B)", fontsize=8.5)
ax.legend(loc="lower right", frameon=False, fontsize=6.5)
fig.tight_layout()
OUT = REPO_ROOT / "figs" / "external_string_copy.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("Wrote external_vowel_tagging, external_procbench, external_string_copy")
