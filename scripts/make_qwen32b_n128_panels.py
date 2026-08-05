#!/usr/bin/env python3
"""Builds an updated Qwen2.5-32B-Instruct-only panel (TAR row + SAR row,
one column per sequence length, now including N=128) for each of the 6
per-task cards that only had the standard grid before. Matches the
established per-task-card visual convention (stars = Base, dashed/dotted
line = naive chance, colored lines per axis value) already used
throughout the report -- only the Qwen-32B panel is rebuilt; the other 4
models' existing images are untouched, since N=128 was only ever run for
Qwen-32B this session.
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

OUT_DIR = REPO_ROOT / "figs" / "png"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL = "Qwen2.5-32B-Instruct"
LENGTHS = [4, 8, 16, 32, 64, 128]
COLORS = {2: "#1b9e77", 4: "#d95f02", 8: "#7570b3", None: "#2166ac"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
})

# (task, axis_col, axis_values, out_suffix)
TASKS = [
    ("random_permutation", "Alphabet_Size", [2, 4, 8]),
    ("conditional_shift", "Alphabet_Size", [2, 4, 8]),
    ("copy_previous", None, [None]),
    ("running_xor", None, [None]),
    ("running_sum_local", "Modulus", [2, 4, 8]),
    ("sliding_window_max", "Alphabet_Size", [4, 8]),
]


def naive_chance_skewed(df, alphabet_size):
    """Matches sliding_window_max's own established convention: fraction of
    ground-truth digits equal to the single most common digit, not naive
    1/alphabet, since this task's output distribution is skewed."""
    sub = df[(df.Method == "Base") & (df.Alphabet_Size == alphabet_size)]
    chars = "".join(sub.Ground_Truth.astype(str))
    if not chars:
        return np.nan
    counts = pd.Series(list(chars)).value_counts()
    return counts.iloc[0] / len(chars)


def load_merged(task):
    grid_dir = REPO_ROOT / "results_clean" / task / "grid"
    frames = [pd.read_csv(p) for p in grid_dir.glob(f"{MODEL}*.csv")]
    df = pd.concat(frames, ignore_index=True)
    key_cols = [c for c in ["Method", "Interval", "Seq_Length", "Alphabet_Size", "Modulus", "Trial"] if c in df.columns]
    df = df.drop_duplicates(subset=key_cols)
    return df[df.Interval != 3]


def make_panel(task, axis_col, axis_values):
    df = load_merged(task)
    lengths = [N for N in LENGTHS if N in df.Seq_Length.unique()]

    fig, axes = plt.subplots(2, len(lengths), figsize=(2.6 * len(lengths), 4.6), sharex=True)
    if len(lengths) == 1:
        axes = axes.reshape(2, 1)

    for col, N in enumerate(lengths):
        for row, metric in enumerate(["TAR", "SAR"]):
            ax = axes[row, col]
            for a in axis_values:
                color = COLORS.get(a, "#2166ac")
                sub = df[df.Seq_Length == N]
                if axis_col:
                    sub = sub[sub[axis_col] == a]

                base = sub[sub.Method == "Base"][metric]
                if len(base):
                    ax.errorbar(0, base.mean(), yerr=base.sem(), color=color,
                                marker="*", markersize=10, capsize=2, linestyle="none", zorder=5)

                bnd = sub[sub.Method == "Cellular_Bounds"]
                if len(bnd):
                    g = bnd.groupby("Interval")[metric].agg(["mean", "sem"]).sort_index()
                    density = 1.0 / g.index.values
                    order = np.argsort(density)
                    label = None
                    if row == 0 and col == 0:
                        label = f"Alphabet={a} (Base: {base.mean():.3f})" if a is not None else f"Base: {base.mean():.3f}"
                    ax.errorbar(density[order], g["mean"].values[order], yerr=g["sem"].values[order],
                                color=color, marker="o", markersize=4, linewidth=1.5, capsize=2, label=label)

                if a is not None:
                    if task == "sliding_window_max":
                        chance = naive_chance_skewed(df, a)
                    else:
                        chance = 1.0 / a
                    if not np.isnan(chance):
                        ax.axhline(chance, color=color, linestyle=":", linewidth=1, alpha=0.6)
                else:
                    ax.axhline(0.5, color=color, linestyle=":", linewidth=1, alpha=0.6)

            ax.set_ylim(0, 1.05)
            ax.set_xlim(-0.05, 1.05)
            ax.set_title(f"{metric}, N={N}", fontsize=8)
            if row == 1:
                ax.set_xlabel("Boundary Density (1/k)")
            if col == 0:
                ax.set_ylabel(metric)

    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=6)
    fig.suptitle(
        f"{task} -- Qwen2.5-32B-Instruct (20 trials/point, now including N=128)\n"
        f"Stars = Base (no boundaries). Dotted lines = naive chance.",
        fontsize=8.5, y=1.03,
    )
    fig.tight_layout()

    out = OUT_DIR / f"{task}_{MODEL}_n128.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main():
    for task, axis_col, axis_values in TASKS:
        make_panel(task, axis_col, axis_values)


if __name__ == "__main__":
    main()
