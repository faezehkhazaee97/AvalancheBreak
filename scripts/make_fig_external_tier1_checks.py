#!/usr/bin/env python3
"""Figures for the "External benchmark tasks" section: vowel/consonant
tagging (7B + 32B), ProcBench-Substitute (7B + 32B), string copying
(7B + 32B). All three now extended to N=128 with full 32B coverage.

Redesigned from the first version: that version plotted Cellular_Bounds
as a single "mean over k" line, which pools away exactly the
best-density effect the accompanying tables highlight (a real
inconsistency -- the vowel_tagging table reports "Best (k)", not a mean).
This version plots each tested density k as its own line (matching the
dose-response philosophy used throughout the rest of this report:
Figure 2, Figure A4), plus a random-chance reference line, instead of
collapsing the density axis into one number.

ProcBench-Substitute and string copying now read from results_clean/
(tasks.run_task's shared pipeline -- the same LENGTHS extended to include
128 in tasks/procbench_substitute/engine.py and tasks/string_copy/
engine.py) rather than the older results/external_tasks/ location, which
tops out at N=64 and 7B-only. TAR values from the two sources are in the
same ballpark (spot-checked before switching) -- the difference is
coverage (N range, 32B), not methodology.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, set_power_of_2_xticks

set_rc()

SEQ_BLUE_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]  # light->dark, sparse->dense


def plot_by_density(ax, df, seqlen_col, method_col, interval_col, tar_col, chance, title, k_labels=None):
    base = df[df[method_col] == "Base"].groupby(seqlen_col)[tar_col].mean()
    ax.axhline(chance, color="#999999", linestyle=":", linewidth=1, alpha=0.85, label="Random chance")
    ax.plot(base.index, base.values, color="black", linestyle="--", marker="o", markersize=4, linewidth=1.3, label="Base")
    ks = sorted(df[df[method_col] != "Base"][interval_col].unique())
    for color, k in zip(SEQ_BLUE_ORDINAL, ks):
        sub = df[df[interval_col] == k].groupby(seqlen_col)[tar_col].mean()
        label = f"k={int(k)}" if k_labels is None else k_labels.get(k, f"k={int(k)}")
        ax.plot(sub.index, sub.values, color=color, marker="s", markersize=3.5, linewidth=1.3, label=label)
    set_power_of_2_xticks(ax, base.index, fontsize=6.5)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("TAR")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=8.5)


# ---- vowel/consonant tagging: 7B vs 32B, one panel each, chance=0.5 (binary V/C) ----
fig, axes = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH * 0.75, 3.0))
for ax, model in zip(axes, ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct"]):
    df = pd.read_csv(REPO_ROOT / "results" / model / "vowel_tagging_boundary_sweep.csv")
    df = df.rename(columns={"Interval": "K"})
    df["Method"] = df["K"].apply(lambda k: "Base" if k == 0 else "Cellular_Bounds")
    plot_by_density(ax, df, "Seq_Length", "Method", "K", "TAR", chance=0.5, title=model.replace("-Instruct", ""))
axes[0].legend(loc="upper right", frameon=False, fontsize=5.5, ncol=1)
fig.suptitle("vowel/consonant tagging: by density k, Base vs. random chance", fontsize=9, y=1.03)
fig.tight_layout()
OUT = REPO_ROOT / "figs" / "external_vowel_tagging.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- ProcBench-Substitute: 7B vs 32B, chance=1/36 (36-symbol native alphabet) ----
fig, axes = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH * 0.75, 3.0))
for ax, model in zip(axes, ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct"]):
    df = pd.read_csv(REPO_ROOT / "results_clean" / "procbench_substitute" / "grid" / f"{model}.csv")
    df = df[df.Interval != 3]
    plot_by_density(ax, df, "Seq_Length", "Method", "Interval", "TAR", chance=1 / 36, title=model.replace("-Instruct", ""))
axes[0].legend(loc="upper right", frameon=False, fontsize=5.5, ncol=1)
fig.suptitle("ProcBench-Substitute: by density k, Base vs. random chance (1/36)", fontsize=9, y=1.03)
fig.tight_layout()
OUT = REPO_ROOT / "figs" / "external_procbench.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- String copying: 7B vs 32B, pooled across alphabet (2/4/8) for the density-line view;
#      chance line uses the mean chance across the three tested alphabets ----
fig, axes = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH * 0.75, 3.0))
mean_chance = sum(1 / a for a in [2, 4, 8]) / 3
for ax, model in zip(axes, ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct"]):
    path = REPO_ROOT / "results_clean" / "string_copy" / "grid" / f"{model}.csv"
    if not path.exists():
        ax.axis("off")
        continue
    df = pd.read_csv(path)
    df = df[df.Interval != 3]
    plot_by_density(ax, df, "Seq_Length", "Method", "Interval", "TAR", chance=mean_chance, title=model.replace("-Instruct", ""))
axes[0].legend(loc="lower left", frameon=False, fontsize=5.5, ncol=1)
fig.suptitle("String copying: by density k, Base vs. random chance (pooled across alphabet 2/4/8)", fontsize=9, y=1.03)
fig.tight_layout()
OUT = REPO_ROOT / "figs" / "external_string_copy.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("Wrote external_vowel_tagging, external_procbench, external_string_copy")
