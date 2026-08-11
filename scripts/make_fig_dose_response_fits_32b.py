#!/usr/bin/env python3
"""Figure rebuild: dose-response scatter with fitted curves, Qwen2.5-32B-
Instruct (was Qwen-7B) -- matches the main paper's single-model story.
Now also plots the bootstrapped vertex confidence interval (previously
only reported in the robustness table's text, never shown visually).
Raw TAR-vs-density point cloud, fitted quadratic, marked vertex + 95% CI,
faceted by sequence length, cyclic_shift alphabet=2.
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
    build_curve, fit_and_score, quadratic, vertex_of_quadratic, bootstrap_vertex_ci,
)
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, power_of_2_label

set_rc()

TASK = "cyclic_shift"
MODEL = "Qwen2.5-32B-Instruct"
AXIS_VAL = 2.0

model_files = select_canonical_files(TASK)
grid_dir = REPO_ROOT / "results_clean" / TASK / "grid"
pooled = grid_dir / f"{MODEL}_POOLED_40trials.csv"
paths = [pooled] if pooled.exists() else model_files[MODEL]
print(f"Provenance: {[str(p.relative_to(REPO_ROOT)) for p in paths]}")

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
            ax.plot(xs, quadratic(xs, a, b, c), color="#eb6834", linewidth=1.5, label="quadratic fit")
            vertex = vertex_of_quadratic(a, b, c)
            if vertex is not None and 0 <= vertex <= 1:
                v_y = quadratic(vertex, a, b, c)
                ax.axvline(vertex, color="#eb6834", linestyle=":", linewidth=1, alpha=0.6)
                ax.plot(vertex, v_y, marker="*", color="#eb6834", markersize=10, zorder=4)
                ci_lo, ci_hi = bootstrap_vertex_ci(x, y)
                if ci_lo is not None:
                    ax.axvspan(max(ci_lo, 0), min(ci_hi, 1), color="#eb6834", alpha=0.12,
                               label="95% vertex CI" if i == 0 else None)

    ax.set_title(f"N={N}", fontsize=8)
    ax.set_xlabel("density (1/k)")
    xtick_vals = sorted(x_raw)
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(["0" if v == 0 else power_of_2_label(v) for v in xtick_vals], fontsize=6, rotation=45)
    if i % 4 == 0:
        ax.set_ylabel("TAR")
    ax.set_ylim(0, 1.05)

axes[-1].axis("off")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.05), frameon=False, fontsize=7)
fig.suptitle("cyclic_shift (alphabet=2, Qwen2.5-32B-Instruct): dose-response curves,\nfitted vertex, and 95% bootstrapped CI, by length", y=1.05)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "dose_response_fits_32b.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
