#!/usr/bin/env python3
"""Analysis of scripts/run_delimiter_length.py (Qwen2.5-32B; markers "|", "," and line break;
N in {64, 128, 256}; k in {4, 16}).

For each task x N x marker (pooled over k):
  pooled TAR with markers vs without; for the position-local control, the share of the
  remaining errors removed, (Bounds - Base) / (1 - Base);
  per position (index scoring): post-marker positions (i > 0, i mod k = 0) and within-segment
  positions, each minus the no-marker run's accuracy at the same indices (mean over the 20
  no-marker trials of that N).
Also reports: outputs shorter than the truth (possible budget cut-offs), and whether the "|"
cells at N = 64 reproduce the main grid exactly (same prompts, greedy decoding).

Outputs: results_clean/_summary/delimiter_length.csv, paper_tmlr/tables/delimiter.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "delimiter_length" / "Qwen2.5-32B-Instruct.csv"
TASKS = [("random_permutation_digits", "random\\_permutation (digits)"), ("running_xor", "running\\_xor"),
         ("running_sum_local", "running\\_sum\\_local")]
MARKERS = [("bar", "\\texttt{|}"), ("comma", "\\texttt{,}"), ("newline", "line break")]
TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}


def vec(truth, pred):
    pred = "" if pd.isna(pred) else str(pred)
    return np.array([i < len(pred) and pred[i] == c for i, c in enumerate(truth)], dtype=float)


def main():
    d = pd.read_csv(SRC, dtype=TEXT, keep_default_na=False)
    rows = []
    for task, _ in TASKS:
        for N in (64, 128, 256):
            x = d[(d.Task == task) & (d.Seq_Length == N)]
            base = x[x.Marker == "base"]
            ref = np.mean([vec(r.Ground_Truth, r.LLM_Output) for r in base.itertuples()], axis=0)
            b_tar = base.TAR.mean()
            for m, _ in MARKERS:
                y = x[x.Marker == m]
                post, within, bpost, bwithin = [], [], [], []
                for r in y.itertuples():
                    k = int(r.Interval)
                    v = vec(r.Ground_Truth, r.LLM_Output)
                    idx = np.arange(len(v))
                    p = (idx > 0) & (idx % k == 0)
                    post.append(v[p].mean()); within.append(v[~p].mean())
                    bpost.append(ref[p].mean()); bwithin.append(ref[~p].mean())
                short = (y.LLM_Output.str.len() < N).mean()
                rows.append(dict(task=task, N=N, marker=m, base=round(b_tar, 3), bounds=round(y.TAR.mean(), 3),
                                 delta=round(y.TAR.mean() - b_tar, 3),
                                 share=round((y.TAR.mean() - b_tar) / (1 - b_tar), 3),
                                 post_gain=round(np.mean(post) - np.mean(bpost), 3),
                                 within_gain=round(np.mean(within) - np.mean(bwithin), 3),
                                 post=round(np.mean(post), 3), short_outputs=round(short, 3),
                                 base_short=round((base.LLM_Output.str.len() < N).mean(), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results_clean" / "_summary" / "delimiter_length.csv", index=False)
    print(out.to_string(index=False))

    # replication of the main grid at N = 64 ("|")
    for task, axis_col, axis in [("running_xor", None, None), ("running_sum_local", "Modulus", 4),
                                 ("random_permutation_digits", "Alphabet_Size", 8)]:
        files = [f for f in (REPO / "results_clean" / task / "grid").glob("Qwen2.5-32B-Instruct*.csv")
                 if "POOLED" not in f.name]
        g = pd.concat([pd.read_csv(f, dtype=TEXT, keep_default_na=False) for f in files])
        g = g[(g.Seq_Length.astype(int) == 64) & g.Interval.astype(int).isin([0, 4, 16])]
        if axis_col:
            g = g[g[axis_col].astype(int) == axis]
        g = g.drop_duplicates(["Method", "Interval", "Trial"]).set_index(["Interval", "Trial"]).LLM_Output
        y = d[(d.Task == task) & (d.Seq_Length == 64) & d.Marker.isin(["base", "bar"])]
        y = y.set_index([y.Interval.astype(int), y.Trial.astype(int)]).LLM_Output
        g.index = g.index.set_levels([g.index.levels[0].astype(int), g.index.levels[1].astype(int)])
        common = y.index.intersection(g.index)
        same = (y.loc[common] == g.loc[common]).mean()
        print(f"replication {task} N=64: {len(common)} matched generations, identical output in {same:.0%}")

    # LaTeX table
    lines = ["\\begin{tabular}{ll ccc ccc ccc}", "\\toprule",
             "& & \\multicolumn{3}{c}{Change in accuracy} & \\multicolumn{3}{c}{Post-marker} & \\multicolumn{3}{c}{Within-segment} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
             "Task & $N$ & " + " & ".join([lab for _, lab in MARKERS] * 3) + " \\\\", "\\midrule"]
    for task, tlab in TASKS:
        for N in (64, 128, 256):
            s = out[(out.task == task) & (out.N == N)].set_index("marker")
            cells = [f"${s.loc[m, c]:+.2f}$" for c in ("delta", "post_gain", "within_gain") for m, _ in MARKERS]
            first = f"\\task{{{tlab}}}" if N == 64 else ""
            lines.append(f"{first} & {N} & " + " & ".join(cells) + " \\\\")
        if task != TASKS[-1][0]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "delimiter.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/delimiter.tex")


if __name__ == "__main__":
    main()
