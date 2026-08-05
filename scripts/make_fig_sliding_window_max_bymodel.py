#!/usr/bin/env python3
"""Per-model sliding_window_max figures, matching the established
per-task-card visual convention used throughout boundaries_report.html
(Fig. 1 cyclic_shift, Fig. 6 running_xor, Fig. 7 running_sum_local, Fig. 8
copy_previous): one PNG per model, TAR on top row / SAR on bottom row,
one panel per sequence length N, x-axis = boundary density (1/k, k=3
excluded), Base plotted as a star at density=0, Cellular_Bounds as a
colored line with error bars across k in {1,2,4,8,16}, and a dashed
horizontal line marking each alphabet size's naive fixed-guess chance
level (frequency of the single most common ground-truth digit -- matches
the same "naive baseline" convention used to screen this task, since
sliding_window_max's output distribution is skewed toward high digits).
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

GRID_DIR = REPO_ROOT / "results_clean" / "sliding_window_max" / "grid"
OUT_DIR = REPO_ROOT / "figs" / "png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LENGTHS = [4, 8, 16, 32, 64]
ALPHABETS = [4, 8]
COLORS = {4: "#2166ac", 8: "#238b45"}  # blue, green -- matches the report's existing palette

MODEL_STATUS = {
    "Qwen2.5-7B-Instruct": "20 trials/point",
    "Qwen2.5-14B-Instruct": "20 trials/point",
    "Qwen2.5-32B-Instruct": "20 trials/point",
    "Mistral-7B-Instruct-v0.3": "20 trials/point",
    "Phi-3.5-mini-instruct": "20 trials/point",
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
})


def naive_chance(df, alphabet_size):
    """Fraction of ground-truth digits equal to the single most common
    digit, pooled across all Base rows at this alphabet size -- the same
    naive fixed-guess baseline used to screen/validate this task."""
    sub = df[(df.Method == "Base") & (df.Alphabet_Size == alphabet_size)]
    chars = "".join(sub.Ground_Truth.astype(str))
    if not chars:
        return np.nan
    counts = pd.Series(list(chars)).value_counts()
    return counts.iloc[0] / len(chars)


def make_one_model(model):
    f = GRID_DIR / f"{model}.csv"
    if not f.exists():
        print(f"SKIP {model}: no grid file")
        return None
    df = pd.read_csv(f)
    df = df[df.Interval != 3]

    chance = {a: naive_chance(df, a) for a in ALPHABETS}

    fig, axes = plt.subplots(2, len(LENGTHS), figsize=(14, 4.6), sharex=True)

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
                                label=f"Alphabet={a} (Base: {base.mean():.3f})" if (row == 0 and col == 0) else None)

                if not np.isnan(chance[a]):
                    ax.axhline(chance[a], color=color, linestyle=":", linewidth=1, alpha=0.6)

            ax.set_ylim(0, 1.05)
            ax.set_xlim(-0.05, 1.05)
            if row == 0:
                ax.set_title(f"{metric}, N={N}", fontsize=8)
            else:
                ax.set_title(f"{metric}, N={N}", fontsize=8)
                ax.set_xlabel("Boundary Density (1/k)")
            if col == 0:
                ax.set_ylabel(metric)

    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=6)
    fig.suptitle(
        f"sliding_window_max (window=3) -- {MODEL_LABELS.get(model, model)} "
        f"({MODEL_STATUS.get(model, '')})\n"
        f"Stars = Base (no boundaries), per alphabet size. Dotted lines = that alphabet's naive fixed-guess chance.",
        fontsize=8.5, y=1.03,
    )
    fig.tight_layout()

    out = OUT_DIR / f"sliding_window_max_{model}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main():
    for model in MODEL_ORDER:
        make_one_model(model)


if __name__ == "__main__":
    main()
