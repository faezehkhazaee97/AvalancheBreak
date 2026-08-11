#!/usr/bin/env python3
"""Figure 1: Tier-1 headline grid (figs/tier1_cyclic_shift.pdf). cyclic_shift,
one panel per model, all three alphabet sizes per panel, TAR vs. sequence
length, Base (black dashed) vs. Cellular_Bounds (colored solid, pooled
across boundary densities k=1/2/4/8/16, k=3 excluded per study convention),
shaded band = +-SEM. Replaces old Figure 1/Figure 2.

Uses select_canonical_files() (DOCUMENTED_MODELS-filtered, POOLED_40trials-
preferring) -- the same canonical-file selection already used throughout
this session's corrected analyses -- so this figure is built from the same
decontaminated data as the FDR/dose-response/completion-rate reruns.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.compute_diagnostics import select_canonical_files
from tasks.common.figure_style import (
    set_rc, savefig, AXIS_COLORS, BASE_STYLE, BOUNDS_STYLE,
    MODEL_ORDER, MODEL_LABELS, MULTI_COL_WIDTH, set_power_of_2_xticks,
)

set_rc()

TASK = "cyclic_shift"
model_files = select_canonical_files(TASK)

# select_canonical_files() always skips *_POOLED_40trials.csv (it lacks
# LLM_Output, needed by row-level diagnostics consumers) and falls back to
# raw _trials0-19/_trials20-39 fragments. For cyclic_shift's 3 Qwen models
# specifically, only a _trials0-19 fragment exists standalone -- the
# second 20 trials (tagged "new_engine" in the POOLED file's own `batch`
# column, alongside a "legacy" first-20 source -- NOT the study's usual
# non-overlapping-seed second-batch convention; flagged separately as a
# provenance question worth its own look) live ONLY inside the POOLED
# file. This figure only needs TAR, not LLM_Output, so it's safe here to
# use the POOLED file directly for the full 40 trials where available --
# this override is local to this figure script, NOT a change to
# select_canonical_files() itself (which stays fragment-preferring for
# every other, LLM_Output-dependent consumer in this repo).
grid_dir = REPO_ROOT / "results_clean" / TASK / "grid"
for model in list(model_files.keys()):
    pooled = grid_dir / f"{model}_POOLED_40trials.csv"
    if pooled.exists():
        model_files[model] = [pooled]

print("Provenance -- canonical files used for Figure 1:")
for m, paths in model_files.items():
    print(f"  {m}: {[str(p.relative_to(REPO_ROOT)) for p in paths]}")

fig, axes = plt.subplots(2, 3, figsize=(MULTI_COL_WIDTH, 4.2), sharey=True)
axes = axes.flatten()

for i, model in enumerate(MODEL_ORDER):
    ax = axes[i]
    if model not in model_files:
        ax.axis("off")
        continue
    frames = [pd.read_csv(p) for p in model_files[model]]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    key_cols = [c for c in ["Method", "Interval", "Seq_Length", "Alphabet_Size", "Trial"] if c in df.columns]
    df = df.drop_duplicates(subset=key_cols)
    df = df[(df.Interval != 3) & (df.Seq_Length <= 128)]  # drop N=256

    base_style_no_color = {k: v for k, v in BASE_STYLE.items() if k != "color"}
    for axis_val in sorted(df.Alphabet_Size.unique()):
        sub = df[df.Alphabet_Size == axis_val]
        base = sub[sub.Method == "Base"].groupby("Seq_Length")["TAR"].agg(["mean", "sem"])
        bnd = sub[sub.Method == "Cellular_Bounds"].groupby("Seq_Length")["TAR"].agg(["mean", "sem"])
        color = AXIS_COLORS.get(axis_val, "gray")

        # Base colored to match its own alphabet (not plain black for all
        # three) so the three dashed lines are distinguishable from each
        # other, while staying visually distinct from Cellular_Bounds via
        # linestyle/marker (dashed circle vs. solid square) alone. No
        # per-panel label here -- one shared legend for the whole figure,
        # built separately below, instead of repeating 6 entries x 5 panels.
        ax.plot(base.index, base["mean"], color=color, **base_style_no_color)
        ax.fill_between(base.index, base["mean"] - base["sem"], base["mean"] + base["sem"],
                         color=color, alpha=0.08)

        ax.plot(bnd.index, bnd["mean"], color=color, **BOUNDS_STYLE)
        ax.fill_between(bnd.index, bnd["mean"] - bnd["sem"], bnd["mean"] + bnd["sem"],
                         color=color, alpha=0.15)

    set_power_of_2_xticks(ax, [4, 8, 16, 32, 64, 128], fontsize=6)
    ax.set_title(MODEL_LABELS[model])
    ax.set_xlabel("Sequence length N")
    if i % 3 == 0:
        ax.set_ylabel("TAR")
    ax.set_ylim(0, 1.05)

axes[5].axis("off")

# One shared legend for the whole figure, split into its two separate
# encodings (color = alphabet size, linestyle/marker = Base vs Bounds)
# instead of 6 redundant "Base, A=2" / "Bounds, A=2" ... combined entries.
color_handles = [Line2D([0], [0], color=AXIS_COLORS[a], linewidth=2, label=f"A={int(a)}")
                  for a in sorted(AXIS_COLORS.keys())]
style_handles = [
    Line2D([0], [0], color="black", linestyle=BASE_STYLE["linestyle"], marker=BASE_STYLE.get("marker", "o"),
           markersize=4, label="Base"),
    Line2D([0], [0], color="black", linestyle=BOUNDS_STYLE["linestyle"], marker=BOUNDS_STYLE.get("marker", "s"),
           markersize=4, label="Cellular_Bounds"),
]
fig.legend(handles=color_handles + style_handles, loc="upper center",
           bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=7.5)

fig.suptitle("cyclic_shift: Base (dashed) vs. Cellular_Bounds (solid), colored by alphabet size", y=1.06)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "tier1_cyclic_shift.pdf"
OUT.parent.mkdir(exist_ok=True)
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
