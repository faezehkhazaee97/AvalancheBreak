#!/usr/bin/env python3
"""Per-model cyclic_shift figures, matching the established per-task-card
visual convention used throughout legacy/boundaries_report.html (same convention
as make_fig_sliding_window_max_bymodel.py: one PNG per model, TAR on top
row / SAR on bottom row, one panel per sequence length N, x-axis =
boundary density (1/k, k=3 excluded), Base plotted as a star at
density=0, Cellular_Bounds as a colored line with error bars across k in
{1,2,4,8,16}, dashed horizontal line marking each alphabet size's random
chance (1/alphabet_size, uniform guess -- cyclic_shift's output is not
digit-skewed like sliding_window_max, so the naive-fixed-guess baseline
used there does not apply here).

Uses the same POOLED_40trials-preferring file selection as
make_fig_tier1_cyclic_shift.py so trial counts match that figure and the
main dose-response fits; restricted to MODEL_ORDER (5 documented models --
see tasks/common/compute_diagnostics.py for why Qwen2.5-72B, an
undocumented, 2-of-9-tasks-only preliminary run, is excluded).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import MODEL_ORDER, MODEL_LABELS

GRID_DIR = REPO_ROOT / "results_clean" / "cyclic_shift" / "grid"
OUT_DIR = REPO_ROOT / "figs" / "png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LENGTHS = [4, 8, 16, 32, 64, 128]
ALPHABETS = [2, 4, 8]
COLORS = {2: "#2166ac", 4: "#238b45", 8: "#d6604d"}

MODEL_STATUS = {
    "Qwen2.5-7B-Instruct": "40 trials/point, 2 independent seed batches",
    "Qwen2.5-14B-Instruct": "40 trials/point, 2 independent seed batches",
    "Qwen2.5-32B-Instruct": "40 trials/point, 2 independent seed batches",
    "Mistral-7B-Instruct-v0.3": "40 trials/point, 2 independent seed batches",
    "Phi-3.5-mini-instruct": "20 trials/point",
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
})


def load_model_df(model):
    pooled = GRID_DIR / f"{model}_POOLED_40trials.csv"
    if pooled.exists():
        return pd.read_csv(pooled)
    plain = GRID_DIR / f"{model}.csv"
    if plain.exists():
        return pd.read_csv(plain)
    return None


def make_one_model(model):
    df = load_model_df(model)
    if df is None:
        print(f"SKIP {model}: no grid file")
        return None
    df = df[(df.Interval != 3) & (df.Seq_Length.isin(LENGTHS))]

    fig, axes = plt.subplots(2, len(LENGTHS), figsize=(16.5, 4.6), sharex=True)

    for col, N in enumerate(LENGTHS):
        for row, metric in enumerate(["TAR", "SAR"]):
            ax = axes[row, col]
            for a in ALPHABETS:
                color = COLORS[a]
                sub = df[(df.Alphabet_Size == a) & (df.Seq_Length == N)]

                base = sub[sub.Method == "Base"][metric]
                if len(base):
                    ax.errorbar(0, base.mean(), yerr=base.sem(), color=color,
                                marker="*", markersize=10, capsize=2, linestyle="none", zorder=5)

                bnd = sub[sub.Method == "Cellular_Bounds"]
                if len(bnd):
                    g = bnd.groupby("Interval")[metric].agg(["mean", "sem"]).sort_index()
                    density = 1.0 / g.index.values
                    order = np.argsort(density)
                    ax.errorbar(density[order], g["mean"].values[order], yerr=g["sem"].values[order],
                                color=color, marker="o", markersize=4, linewidth=1.5, capsize=2,
                                label=f"Alphabet={a} (Base: {base.mean():.3f})" if (row == 0 and col == 0 and len(base)) else None)

                ax.axhline(1.0 / a, color=color, linestyle=":", linewidth=1, alpha=0.6)

            ax.set_ylim(0, 1.05)
            ax.set_xlim(-0.05, 1.05)
            ax.set_title(f"{metric}, N={N}", fontsize=8)
            if row == 1:
                ax.set_xlabel("Boundary Density (1/k)")
            if col == 0:
                ax.set_ylabel(metric)

    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=6)
    fig.suptitle(
        f"cyclic_shift (all alphabet sizes) -- {MODEL_LABELS.get(model, model)} "
        f"({MODEL_STATUS.get(model, '')})\n"
        f"Stars = Base (no boundaries), per alphabet size. Dotted lines = that alphabet's random chance.",
        fontsize=8.5, y=1.03,
    )
    fig.tight_layout()

    out = OUT_DIR / f"cyclic_shift_{model}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main():
    for model in MODEL_ORDER:
        make_one_model(model)


if __name__ == "__main__":
    main()
