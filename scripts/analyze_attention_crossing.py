#!/usr/bin/env python3
"""Analysis of scripts/run_attention_crossing.py: attention from the position that predicts answer
character j to the input character j-1 (the one before the marker at post-marker positions), to
the input character j, to the marker, and to the answer just written, by layer, in the clean and
marked teacher-forced runs, at post-marker and within-segment positions.

Outputs: results_clean/_summary/attention_crossing.csv (per model, task, run, position type, layer),
         paper_tmlr/figures/attention_crossing.pdf (attention to the input before the marker, by layer),
         paper_tmlr/tables/attention_crossing.tex (sums over layers and the peak layer)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH  # noqa: E402

MODELS = [("Qwen2.5 32B", "Qwen2.5-32B-Instruct"), ("Gemma 4 31B", "gemma-4-31B-it"), ("Qwen3.8 27B", "Qwen3.8-27B")]
TASKN = {"running_xor": "Running XOR", "running_sum_local": "Running sum (mod 4)"}
COLS = ["prev_in", "cur_in", "marker", "prev_out", "max_head_prev_in"]


def main():
    frames = []
    for label, tag in MODELS:
        f = REPO / "results" / "attention_crossing" / f"{tag}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        trials = d.groupby(["Task", "Run"]).Trial.nunique().min()
        m = d.groupby(["Task", "Run", "PosType", "Layer"])[COLS].mean().reset_index()
        m.insert(0, "model", label); m["n_trials"] = trials
        frames.append(m)
    if not frames:
        print("no results yet"); return
    o = pd.concat(frames, ignore_index=True).round(4)
    # Qwen3.8 is a hybrid stack: 64 layers of which every 4th is full attention
    # (config full_attention_interval = 4), and only those produce attention weights. The runner
    # enumerates what it is given, so its Layer is the softmax index 0..15; map it to the
    # full-stack index 4i+3 so layer numbers mean the same thing in every row of the table.
    hyb = o.model == "Qwen3.8 27B"
    assert not hyb.any() or o.loc[hyb, "Layer"].max() == 15, "unexpected layer count for Qwen3.8"
    o.loc[hyb, "Layer"] = 4 * o.loc[hyb, "Layer"] + 3
    o.to_csv(REPO / "results_clean" / "_summary" / "attention_crossing.csv", index=False)
    s = o.groupby(["model", "Task", "Run", "PosType"])[COLS].sum().round(3)   # mass summed over layers
    pd.set_option("display.width", 200)
    print("attention mass summed over layers (mean over heads):\n", s.to_string())

    set_rc()
    models = [m for m, _ in MODELS if m in set(o.model)]
    fig, axes = plt.subplots(len(models), len(TASKN), figsize=(MULTI_COL_WIDTH, 2.0 * len(models)), squeeze=False)
    styles = {("clean", "post"): ("#009E73", "-", "clean run, same positions"), ("marked", "post"): ("#D55E00", "-", "marked run, post-marker"),
              ("marked", "within"): ("#0072B2", "--", "marked run, within-segment")}
    for i, label in enumerate(models):
        for j, task in enumerate(TASKN):
            ax = axes[i][j]
            for (run, ptype), (col, ls, name) in styles.items():
                x = o[(o.model == label) & (o.Task == task) & (o.Run == run) & (o.PosType == ptype)].sort_values("Layer")
                if not x.empty:
                    ax.plot(x.Layer, x.prev_in, color=col, linestyle=ls, linewidth=1.4, label=name)
            x = o[(o.model == label) & (o.Task == task) & (o.Run == "marked") & (o.PosType == "post")].sort_values("Layer")
            if not x.empty:
                ax.plot(x.Layer, x.marker, color="#CC79A7", linestyle=":", linewidth=1.1, label="marked run, mass on the marker")
            ax.set_title(f"{label}: {TASKN[task]}", loc="left"); ax.set_xlabel("layer")
            ax.set_ylabel("attention to input $j-1$" if j == 0 else "")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); savefig(fig, REPO / "paper_tmlr" / "figures" / "attention_crossing.pdf")

    lines = ["\\begin{tabular}{ll l cccc c}", "\\toprule",
             "Model & Task & Run, positions & Input $j-1$ & Input $j$ & Marker & Answer $j-1$ & Peak layer for input $j-1$ \\\\", "\\midrule"]
    names = {("clean", "post"): "clean, post-marker indices", ("marked", "post"): "marked, post-marker", ("marked", "within"): "marked, within-segment",
             ("input_only", "post"): "input markers only, post-marker"}
    for label in models:
        first = True
        for task in TASKN:
            for key, nm in names.items():
                x = o[(o.model == label) & (o.Task == task) & (o.Run == key[0]) & (o.PosType == key[1])]
                if x.empty:
                    continue
                tot = x[COLS].sum(); peak = int(x.loc[x.prev_in.idxmax()].Layer)
                lines.append(f"{label if first else ''} & {TASKN[task] if key == ('clean', 'post') else ''} & {nm} & {tot.prev_in:.2f} & {tot.cur_in:.2f} & "
                             + (f"{tot.marker:.2f}" if key[0] in ("marked", "input_only") else "--") + f" & {tot.prev_out:.2f} & {peak} \\\\")
                first = False
            lines.append("\\addlinespace")
    while lines[-1] == "\\addlinespace":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "attention_crossing.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/attention_crossing.tex")


if __name__ == "__main__":
    main()
