"""Paper Figure 1: the helps / cancelled / restored split, every task x model.

One cell = pooled TAR change (Cellular_Bounds - Base) for a task and model,
pooled over N in {4,8,16,32,64}, every tested density (k=3 excluded), and
every alphabet/modulus value. Lengths are capped at 64 so every model is
pooled over the same grid (the 32B-only N=128 extension and cyclic_shift's
N=128/256 are left out here). '*' marks |delta / combined SEM| at or above the
Bonferroni threshold across all plotted cells (two-sided p < 0.05; 3.38 for 70 cells).

Three row blocks: position-local tasks, dependency tasks as-is, and the same
dependency tasks with the dependency severed at each marker (reset variants).
Models without data for a cell are left blank. Newly added models
(Qwen3.8-27B, Gemma-4-31B-it) appear automatically once their grid CSVs exist.

Outputs: figs/fig1_split_heatmap.pdf (+ png) and
results_clean/_summary/split_heatmap_values.csv (the numbers quoted in the text).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH  # noqa: E402

LENGTHS = [4, 8, 16, 32, 64]
VMAX = 0.30
SIG_Z = None  # set below to the Bonferroni threshold over the plotted cells

ROW_BLOCKS = [
    ("Position-local", ["cyclic_shift", "random_permutation", "conditional_shift"]),
    ("Dependency, as is", ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]),
    ("Dependency severed\n(reset variant)", ["copy_previous_reset", "running_xor_reset",
                                             "running_sum_local_reset", "sliding_window_max_reset"]),
]
# (column label, local model tag or None for Gemini)
MODELS = [
    ("Qwen2.5\n7B", "Qwen2.5-7B-Instruct"),
    ("Qwen2.5\n14B", "Qwen2.5-14B-Instruct"),
    ("Qwen2.5\n32B", "Qwen2.5-32B-Instruct"),
    ("Mistral\n7B", "Mistral-7B-Instruct-v0.3"),
    ("Phi-3.5\nmini", "Phi-3.5-mini-instruct"),
    ("Qwen3.8\n27B", "Qwen3.8-27B"),
    ("Gemma 4\n31B", "gemma-4-31B-it"),
    ("Gemini\n3.5 Flash", None),
]
GEMINI_CSV = REPO / "results_clean" / "gemini_pilot" / "gemini-flash-latest_POOLED_3trials.csv"

# Diverging pair from the validated palette (blue <-> red, neutral gray midpoint).
CMAP = LinearSegmentedColormap.from_list("helps_hurts", ["#b52f2f", "#f0efec", "#184f95"])
INK, INK_ON_DARK, BLANK = "#0b0b0b", "#ffffff", "#fcfcfb"


def load_local(task, tag):
    # Raw per-generation files only. *_POOLED_* summaries are skipped: for cyclic_shift at
    # Qwen2.5-7B/14B/32B they hold a second, "legacy" batch under the same trial numbers,
    # which sorts first and would silently replace the raw (k=1-budget-corrected) batch.
    files = [f for f in sorted((REPO / "results_clean" / task / "grid").glob(f"{tag}*.csv"))
             if "POOLED" not in f.name]
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    axis = next((c for c in ("Alphabet_Size", "Modulus") if c in df.columns), None)
    keys = ["Method", "Interval", "Seq_Length", "Trial"] + ([axis] if axis else [])
    return df.drop_duplicates(subset=keys)


def load_gemini(task):
    if not GEMINI_CSV.exists():
        return None
    df = pd.read_csv(GEMINI_CSV)
    df = df[df.Task == task]
    return df if len(df) else None


def pooled_delta(df):
    df = df[(df.Interval != 3) & (df.Seq_Length.isin(LENGTHS))]
    base, bnd = df[df.Method == "Base"].TAR, df[df.Method == "Cellular_Bounds"].TAR
    if len(base) < 2 or len(bnd) < 2:
        return None
    delta = bnd.mean() - base.mean()
    z = delta / np.sqrt(base.sem() ** 2 + bnd.sem() ** 2)
    return dict(base=base.mean(), bounds=bnd.mean(), delta=delta, z=z, n_base=len(base), n_bounds=len(bnd))


rows, records = [], []
for block, tasks in ROW_BLOCKS:
    for task in tasks:
        cells = []
        for label, tag in MODELS:
            df = load_gemini(task) if tag is None else load_local(task, tag)
            r = pooled_delta(df) if df is not None else None
            cells.append(r)
            if r:
                records.append(dict(block=block.replace("\n", " "), task=task,
                                    model=label.replace("\n", " "), **r))
        rows.append((block, task, cells))

# Drop model columns with no data at all (e.g. new models before their runs finish).
keep = [j for j in range(len(MODELS)) if any(cells[j] for _, _, cells in rows)]
models = [MODELS[j] for j in keep]
rows = [(b, t, [c[j] for j in keep]) for b, t, c in rows]

from scipy.stats import norm  # noqa: E402
SIG_Z = norm.ppf(1 - 0.05 / (2 * len(records)))  # Bonferroni, two-sided, over plotted cells
print(f"{len(records)} cells; star threshold |z| >= {SIG_Z:.2f}")

out_csv = REPO / "results_clean" / "_summary" / "split_heatmap_values.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).round(4).to_csv(out_csv, index=False)
print(f"Wrote {out_csv}")

set_rc()
n_rows, n_cols = len(rows), len(models)
fig, ax = plt.subplots(figsize=(MULTI_COL_WIDTH * 0.92, 0.27 * n_rows + 1.1))
grid = np.full((n_rows, n_cols), np.nan)
for i, (_, _, cells) in enumerate(rows):
    for j, c in enumerate(cells):
        if c:
            grid[i, j] = c["delta"]

ax.set_facecolor(BLANK)
im = ax.imshow(np.ma.masked_invalid(grid), cmap=CMAP, vmin=-VMAX, vmax=VMAX, aspect="auto")
for i, (_, _, cells) in enumerate(rows):
    for j, c in enumerate(cells):
        if not c:
            ax.text(j, i, "—", ha="center", va="center", fontsize=6.5, color="#9a9a96")
            continue
        txt = f"{c['delta']:+.2f}" + ("*" if abs(c["z"]) >= SIG_Z else "")
        dark = abs(c["delta"]) > 0.55 * VMAX
        ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                color=INK_ON_DARK if dark else INK)

# 2px surface gaps between cells, heavier rules between row blocks.
ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
ax.grid(which="minor", color="#ffffff", linewidth=1.5)
ax.tick_params(which="minor", length=0)
start = 0
for block, tasks in ROW_BLOCKS:
    if start:
        ax.axhline(start - 0.5, color="#0b0b0b", linewidth=0.9)
    ax.text(n_cols - 0.35, start + (len(tasks) - 1) / 2, block, ha="left", va="center",
            fontsize=7, color="#4a4a47")
    start += len(tasks)

ax.set_xticks(range(n_cols))
ax.set_xticklabels([m[0] for m in models], fontsize=6.5)
ax.xaxis.tick_top()
ax.set_yticks(range(n_rows))
ax.set_yticklabels([t.replace("_reset", " (reset)") for _, t, _ in rows], fontsize=7)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)

cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, aspect=40, shrink=0.6)
cb.set_label("TAR change with markers (Cellular_Bounds − Base)   ← hurts    helps →", fontsize=7)
cb.ax.tick_params(labelsize=6.5)
cb.outline.set_visible(False)

out = REPO / "figs" / "fig1_split_heatmap.pdf"
savefig(fig, out)
fig.savefig(REPO / "figs" / "png" / "fig1_split_heatmap.png", dpi=200, bbox_inches="tight")
