#!/usr/bin/env python3
"""Figure 2: Dose-response scatter with fitted curves (figs/dose_response_
fits.pdf). Raw TAR-vs-density point cloud with fitted quadratic curve and
marked vertex, faceted by sequence length, for cyclic_shift (alphabet=2,
Qwen2.5-7B -- the headline SK-fit condition). Companion to the aggregate
R^2 bar chart already in the draft: this shows the actual curves and
vertex locations the aggregate summarizes.

Reuses the exact same fit machinery (build_curve, fit_and_score,
vertex_of_quadratic) as tasks/common/compute_dose_response_robustness.py --
the already-verified, decontaminated 348-row analysis -- so this figure is
guaranteed consistent with those numbers, not a re-derivation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.compute_diagnostics import select_canonical_files
from tasks.common.compute_dose_response_robustness import (
    build_curve, fit_and_score, quadratic, vertex_of_quadratic,
)
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH

set_rc()

TASK = "cyclic_shift"
MODEL = "Qwen2.5-7B-Instruct"
AXIS_VAL = 2.0

model_files = select_canonical_files(TASK)
grid_dir = REPO_ROOT / "results_clean" / TASK / "grid"
pooled = grid_dir / f"{MODEL}_POOLED_40trials.csv"
paths = [pooled] if pooled.exists() else model_files[MODEL]
print(f"Provenance -- canonical file(s) used for Figure 2: {[str(p.relative_to(REPO_ROOT)) for p in paths]}")

frames = [pd.read_csv(p) for p in paths]
df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
key_cols = [c for c in ["Method", "Interval", "Seq_Length", "Alphabet_Size", "Trial"] if c in df.columns]
df = df.drop_duplicates(subset=key_cols)

lengths = sorted(df.Seq_Length.unique())
fig, axes = plt.subplots(2, 4, figsize=(MULTI_COL_WIDTH, 3.6), sharey=True)
axes = axes.flatten()

for i, N in enumerate(lengths):
    ax = axes[i]
    sub = df[(df.Seq_Length == N) & (df.Alphabet_Size == AXIS_VAL)]
    sub = sub[sub.Interval != 3]
    per_k = sub.groupby("Interval")["TAR"]
    x_raw = np.array([1 / k if k > 0 else 0 for k in per_k.groups.keys()])
    y_raw = per_k.mean().values
    ax.scatter(x_raw, y_raw, color="black", s=14, zorder=3, label="observed mean")

    x, y = build_curve(df, N, AXIS_VAL, "Alphabet_Size")
    if len(x) >= 5:
        scores = fit_and_score(x, y)
        if scores and scores.get("quadratic") is not None:
            a, b, c = scores["quadratic"]["params"]
            xs = np.linspace(0, 1, 100)
            ax.plot(xs, quadratic(xs, a, b, c), color="#d95f02", linewidth=1.5, label="quadratic fit")
            vertex = vertex_of_quadratic(a, b, c)
            if vertex is not None and 0 <= vertex <= 1:
                v_y = quadratic(vertex, a, b, c)
                ax.axvline(vertex, color="#d95f02", linestyle=":", linewidth=1, alpha=0.6)
                ax.plot(vertex, v_y, marker="*", color="#d95f02", markersize=10, zorder=4)

    ax.set_title(f"N={N}", fontsize=8)
    ax.set_xlabel("density (1/k)")
    if i % 4 == 0:
        ax.set_ylabel("TAR")
    ax.set_ylim(0, 1.05)

axes[-1].axis("off")
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.05), frameon=False, fontsize=7)
fig.suptitle("cyclic_shift (alphabet=2, Qwen2.5-7B): dose-response curves and fitted vertex, by length", y=1.03)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "dose_response_fits.pdf"
savefig(fig, OUT)
