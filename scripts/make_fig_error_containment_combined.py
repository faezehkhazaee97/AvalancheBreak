#!/usr/bin/env python3
"""Error containment, Qwen-7B and Qwen-32B side by side (figs/error_containment_combined.pdf).
P(next character correct | previous character wrong), split by whether the
next position is immediately after a boundary vs. elsewhere. 2x2 grid:
rows = model (7B top, 32B bottom), columns = task (cyclic_shift left,
modsum right). Both models' data already existed at full scale (N up to
256, including 128) -- no new run was needed, just plotting 7B alongside
the existing 32B panels instead of 32B alone.

Supersedes make_fig_error_containment.py (7B-only) and
make_fig_error_containment_32b.py (32B-only, replaced the 7B version
outright rather than showing both) -- this version keeps both scales in
one figure so the reader can see whether the pattern holds, strengthens,
or changes with scale, instead of only ever seeing one model at a time.

REPRESENTATIVE_AXIS_VAL = 2 (alphabet=2 for cyclic_shift, modulus=2 for
modsum), not pooled across the full {2,4,8} sweep like the previous
version. Checked directly: modulus=4/8 sit on a much lower absolute scale
than modulus=2 (harder task, larger answer space -> lower baseline
recovery rate for everyone), so averaging all three together compresses
the modsum panel into a narrow, muddy band that doesn't clearly show
either line -- pooling across different difficulty levels, not a fair
average. alphabet=2 / modulus=2 matches the "simplest representative
condition" convention already used by every other mechanism figure in
this report (Figure 1, the causal-ablation figure, etc.), rather than
introducing a new convention just for this one.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, set_power_of_2_xticks

set_rc()


def reconstruct_binary_series(ground_truth, llm_output):
    truth = str(ground_truth)
    pred = str(llm_output)[: len(truth)]
    return [1 if i < len(pred) and pred[i] == truth[i] else 0 for i in range(len(truth))]


def is_boundary_adjacent(pos, interval):
    return interval > 0 and pos > 0 and pos % interval == 0


def error_recovery_events(df, axis_col):
    events = []
    for _, row in df.iterrows():
        series = reconstruct_binary_series(row["Ground_Truth"], row["LLM_Output"])
        interval = row["Interval"]
        axis_val = row[axis_col] if axis_col else None
        for i in range(1, len(series)):
            if series[i - 1] == 0:
                events.append({
                    "Method": row["Method"], "Interval": interval, "Seq_Length": row["Seq_Length"],
                    "Axis_Val": axis_val,
                    "Boundary_Adjacent": is_boundary_adjacent(i, interval), "Recovered": series[i],
                })
    return pd.DataFrame(events)


REPRESENTATIVE_AXIS_VAL = 2

MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct"]
TASKS = [("cyclic_shift (alphabet=2)", "thermodynamics_grid_templated_v2.csv", "Alphabet_Size"),
         ("modsum (global full-prefix dependency, modulus=2)", "running_modsum_grid_templated_v2.csv", "Modulus")]

fig, axes = plt.subplots(2, 2, figsize=(MULTI_COL_WIDTH, 6.0))

for row, model in enumerate(MODELS):
    results_dir = REPO_ROOT / "results" / model
    for col, (label, fname, axis_col) in enumerate(TASKS):
        ax = axes[row, col]
        df = pd.read_csv(results_dir / fname)
        df = df[df[axis_col] == REPRESENTATIVE_AXIS_VAL]  # only Cellular_Bounds is plotted below anyway, but restrict Base too for a clean single-condition source
        events = error_recovery_events(df, axis_col)
        cell_events = events[events.Method == "Cellular_Bounds"]
        per_N = cell_events.groupby(["Seq_Length", "Boundary_Adjacent"])["Recovered"].mean().reset_index()
        for adj_val, color, marker, lbl in [(True, "#1b9e77", "o", "Right after boundary"),
                                              (False, "#eb6834", "s", "Elsewhere")]:
            sub = per_N[per_N.Boundary_Adjacent == adj_val].sort_values("Seq_Length")
            if len(sub):
                ax.plot(sub.Seq_Length, sub.Recovered, color=color, marker=marker, markersize=4,
                         linewidth=1.5, label=lbl if (row == 0 and col == 0) else None)
        set_power_of_2_xticks(ax, per_N.Seq_Length.unique(), fontsize=6.5)
        if row == 1:
            ax.set_xlabel("Sequence length N")
        if col == 0:
            ax.set_ylabel(f"{model.replace('-Instruct','')}\nP(next correct | previous wrong)")
        if row == 0:
            ax.set_title(label, fontsize=9)
        ax.set_ylim(0, 1.0)

fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=8)
fig.suptitle("Error containment: recovery immediately after a boundary vs. elsewhere, by model and task", y=1.07, fontsize=9.5)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "error_containment_combined.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("Wrote error_containment_combined")
