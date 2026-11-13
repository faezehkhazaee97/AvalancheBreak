#!/usr/bin/env python3
"""Jittered marker placement on the dependency tasks (reviewer question: how does the blind spot's
magnitude change when markers are not exactly periodic?).

The <task>_jitter runs (tasks/common/jitter.py) move every marker by -1, 0 or +1 positions from
its periodic place at k = 4 and 8, on the same inputs and seeds as the main grid's k = 4 and 8
cells. Positions are classified from where the markers actually are: "post-marker" is the
character right after a marker, "within-segment" every other one. Both the exactly periodic run
(main grid) and the jittered run are compared with the no-marker run at the same indices
(N <= 64; index scoring), so the two columns differ only in placement.

Outputs: results_clean/_summary/jitter_dependency.csv, paper_tmlr/tables/jitter_dependency.tex
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, LENGTHS  # noqa: E402
from analyze_aligned_scoring import index_vector, clean  # noqa: E402

MODELS = [("Qwen2.5 7B", "Qwen2.5-7B-Instruct"), ("Qwen2.5 32B", "Qwen2.5-32B-Instruct")]
TASKS = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]
KS = [4, 8]


def post_mask(shown, n):
    """Boolean mask over plain indices: True where the character follows a marker."""
    m = np.zeros(n, dtype=bool)
    j = 0
    for c in shown:
        if c == "|":
            if j < n:
                m[j] = True
        else:
            j += 1
    m[0] = False
    return m


def compare(base, bnd):
    ref = {}
    for _, r in base.iterrows():
        v = index_vector(clean(r.Ground_Truth), clean(r.LLM_Output)); ref.setdefault(len(v), []).append(v)
    ref = {n: np.mean(vs, axis=0) for n, vs in ref.items()}
    post, bpost, within, bwithin, acc, bacc = [], [], [], [], [], []
    for _, r in bnd.iterrows():
        v = index_vector(clean(r.Ground_Truth), clean(r.LLM_Output))
        if len(v) not in ref:
            continue
        p = post_mask(r.Input_Given, len(v))
        if not p.any():
            continue
        post.append(v[p].mean()); bpost.append(ref[len(v)][p].mean())
        within.append(v[~p].mean()); bwithin.append(ref[len(v)][~p].mean())
        acc.append(v.mean()); bacc.append(ref[len(v)].mean())
    return dict(n=len(post), acc=np.mean(acc), delta=np.mean(acc) - np.mean(bacc), post=np.mean(post),
                post_gain=np.mean(post) - np.mean(bpost), within_gain=np.mean(within) - np.mean(bwithin))


def main():
    rows = []
    for task in TASKS:
        for lab, tag in MODELS:
            d, j = load(task, tag), load(task + "_jitter", tag)
            if d is None or j is None:
                continue
            d = d[d.Seq_Length.astype(int).isin(LENGTHS)]
            j = j[j.Seq_Length.astype(int).isin(LENGTHS)]
            base = d[d.Method == "Base"]
            for name, src in [("periodic", d), ("jittered", j)]:
                bnd = src[(src.Method == "Cellular_Bounds") & src.Interval.astype(int).isin(KS)]
                if bnd.empty:
                    continue
                rows.append(dict(task=task, model=lab, placement=name, **compare(base, bnd)))
    o = pd.DataFrame(rows).round(4)
    if o.empty:
        print("no results yet"); return
    o.to_csv(REPO / "results_clean" / "_summary" / "jitter_dependency.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.to_string(index=False))

    P = ["periodic", "jittered"]
    lines = ["\\begin{tabular}{ll cc cc cc}", "\\toprule",
             "& & \\multicolumn{2}{c}{Pooled $\\Delta$} & \\multicolumn{2}{c}{Post-marker} & \\multicolumn{2}{c}{Within-segment} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
             "Task & Model & Periodic & Jittered & Periodic & Jittered & Periodic & Jittered \\\\", "\\midrule"]
    for task in TASKS:
        first = True
        for lab, _ in MODELS:
            x = o[(o.task == task) & (o.model == lab)].set_index("placement")
            if len(x) < 2:
                continue
            c = [f"${x.loc[p, 'delta']:+.2f}$" for p in P] + [f"${x.loc[p, 'post_gain']:+.2f}$" for p in P] + [f"${x.loc[p, 'within_gain']:+.2f}$" for p in P]
            lines.append(("\\task{" + task.replace("_", "\\_") + "}" if first else "") + f" & {lab} & " + " & ".join(c) + " \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "jitter_dependency.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/jitter_dependency.tex")


if __name__ == "__main__":
    main()
