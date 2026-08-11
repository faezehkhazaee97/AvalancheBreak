#!/usr/bin/env python3
"""Failure-mode diagnostics, by-model breakdown figures. Replaces an
earlier version of these figures that pooled across sequence length
within each alphabet x method cell -- that pooling hid a real,
length-dependent sign flip in cyclic_shift's WFR (boundaries improve WFR
at N=16/32, hurt it at N=64+; the pooled number showed only "hurts").
This version puts N on the x-axis directly (unpooled) and fixes the
secondary axis (alphabet size / modulus) to one representative value per
task instead of averaging over it -- the same "pick one clean value,
don't average" convention Figures 1 and 3 already use. Also drops
Qwen2.5-72B, an undocumented, never-written-up model that had leaked
into the earlier version of these figures; the 5-model roster here
matches every other cross-model comparison in this report.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, MODEL_ORDER, MODEL_LABELS, set_power_of_2_xticks

set_rc()

DIAG_DIR = REPO_ROOT / "results_clean" / "_diagnostics"

CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#a35fd6", "#c9962e"]
MODEL_COLOR = {m: c for m, c in zip(MODEL_ORDER, CAT_COLORS)}
MODEL_MARKER = {m: mk for m, mk in zip(MODEL_ORDER, ["o", "s", "^", "D", "v"])}

# (task, axis_col, axis_val, title_note)
TASKS = [
    ("cyclic_shift", "Alphabet_Size", 2, "alphabet=2"),
    ("random_permutation", "Alphabet_Size", 2, "alphabet=2"),
    ("conditional_shift", "Alphabet_Size", 2, "alphabet=2"),
    ("running_xor", None, None, "binary"),
    ("running_sum_local", "Modulus", 4, "modulus=4"),
    ("copy_previous", None, None, "binary"),
    ("sliding_window_max", "Alphabet_Size", 8, "alphabet=8"),
]


def make_task_fig(task, axis_col, axis_val, note):
    fig, axes = plt.subplots(2, 2, figsize=(MULTI_COL_WIDTH * 0.62, 4.6), sharex=True)
    any_model_plotted = False
    all_N = set()

    for model in MODEL_ORDER:
        row_path = DIAG_DIR / f"{task}__{model}__rows.csv"
        if not row_path.exists():
            continue
        df = pd.read_csv(row_path)
        if axis_col:
            if axis_col not in df.columns:
                continue
            df = df[df[axis_col] == axis_val]
        if len(df) == 0:
            continue
        any_model_plotted = True
        color, marker = MODEL_COLOR[model], MODEL_MARKER[model]
        label = MODEL_LABELS[model]

        for col_idx, method in enumerate(["Base", "Cellular_Bounds"]):
            sub = df[df.Method == method]
            if len(sub) == 0:
                continue
            g_wfr = sub.groupby("Seq_Length")["Well_Formed"].mean()
            axes[0, col_idx].plot(g_wfr.index, g_wfr.values, color=color, marker=marker,
                                   markersize=4, linewidth=1.4, label=label)

            wf_only = sub[sub.Well_Formed]
            g_tarwf = wf_only.groupby("Seq_Length")["TAR"].mean()
            axes[1, col_idx].plot(g_tarwf.index, g_tarwf.values, color=color, marker=marker,
                                   markersize=4, linewidth=1.4, label=label)
            all_N.update(g_wfr.index)

    if not any_model_plotted:
        plt.close(fig)
        return None

    axes[0, 0].set_title("WFR -- Base (no boundaries)", fontsize=8)
    axes[0, 1].set_title("WFR -- Cellular_Bounds (with boundaries)", fontsize=8)
    axes[1, 0].set_title("TAR | Well-Formed -- Base (no boundaries)", fontsize=8)
    axes[1, 1].set_title("TAR | Well-Formed -- Cellular_Bounds (with boundaries)", fontsize=8)
    for ax in axes[0]:
        ax.set_ylabel("WFR", fontsize=7.5)
    for ax in axes[1]:
        ax.set_ylabel("TAR | Well-Formed", fontsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("Sequence length N", fontsize=7.5)
    for ax in axes.flat:
        ax.set_ylim(0, 1.05)
        set_power_of_2_xticks(ax, all_N, fontsize=6.5)
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=6)

    fig.suptitle(f"{task} -- failure-mode diagnostics by model\nN on x-axis (not pooled), {note}", fontsize=8.5, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


for task, axis_col, axis_val, note in TASKS:
    fig = make_task_fig(task, axis_col, axis_val, note)
    if fig is None:
        print(f"SKIP {task}: no data found")
        continue
    out = REPO_ROOT / "figs" / f"diag_bymodel_{task}.pdf"
    savefig(fig, out)
    fig.savefig(str(out).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
