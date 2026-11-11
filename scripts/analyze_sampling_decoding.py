#!/usr/bin/env python3
"""Analysis of scripts/run_sampling_decoding.py: marker effects under sampling vs greedy.

Greedy reference: the main-grid generations for the same prompts (Qwen2.5-32B, trials 0-19,
N in {32, 64}, k in {0, 4, 16}, random_permutation_digits A = 8, running_sum_local modulus 4).
For each decoding regime and task (pooled over N and k): accuracy without and with markers and
the difference with a 95% interval; for the dependency tasks, post-marker and within-segment
accuracy minus the no-marker run of the same regime at the same positions.

Outputs: results_clean/_summary/sampling_decoding.csv, paper_tmlr/tables/sampling.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}
TASKS = [("random_permutation_digits", "Alphabet_Size", 8, "random\\_permutation (digits)"),
         ("running_xor", None, None, "running\\_xor"), ("running_sum_local", "Modulus", 4, "running\\_sum\\_local")]
REGIMES = [("greedy", "Greedy"), ("t07", "$T = 0.7$, top-$p$ 0.8"), ("t10", "$T = 1.0$, top-$p$ 0.95")]


def vec(truth, pred):
    pred = "" if pd.isna(pred) else str(pred)
    return np.array([i < len(pred) and pred[i] == c for i, c in enumerate(truth)], dtype=float)


def greedy(task, axis_col, axis):
    files = [f for f in (REPO / "results_clean" / task / "grid").glob("Qwen2.5-32B-Instruct*.csv")
             if "POOLED" not in f.name]
    g = pd.concat([pd.read_csv(f, dtype=TEXT, keep_default_na=False) for f in files])
    g = g.assign(N=g.Seq_Length.astype(int), k=g.Interval.astype(int), t=g.Trial.astype(int), TAR=g.TAR.astype(float))
    g = g[g.N.isin([32, 64]) & g.k.isin([0, 4, 16]) & (g.t < 20)]
    if axis_col:
        g = g[g[axis_col].astype(int) == axis]
    return g.drop_duplicates(["k", "N", "t"])


def main():
    s = pd.read_csv(REPO / "results" / "sampling_decoding" / "Qwen2.5-32B-Instruct.csv", dtype=TEXT, keep_default_na=False)
    s = s.assign(N=s.Seq_Length.astype(int), k=s.Interval.astype(int), t=s.Trial.astype(int), TAR=s.TAR.astype(float))
    rows = []
    for task, axis_col, axis, _ in TASKS:
        for reg, _ in REGIMES:
            d = greedy(task, axis_col, axis) if reg == "greedy" else s[(s.Task == task) & (s.Regime == reg)]
            base, bnd = d[d.k == 0], d[d.k > 0]
            # paired by (N, trial): each marker run minus the no-marker run of the same N and trial index
            b = base.set_index(["N", "t"]).TAR
            diff = bnd.TAR.values - b.reindex(list(zip(bnd.N, bnd.t))).values
            ref = {N: np.mean([vec(r.Ground_Truth, r.LLM_Output) for r in base[base.N == N].itertuples()], axis=0)
                   for N in (32, 64)}
            post, within, bpost, bwithin = [], [], [], []
            for r in bnd.itertuples():
                v = vec(r.Ground_Truth, r.LLM_Output)
                idx = np.arange(len(v)); p = (idx > 0) & (idx % r.k == 0)
                post.append(v[p].mean()); within.append(v[~p].mean())
                bpost.append(ref[r.N][p].mean()); bwithin.append(ref[r.N][~p].mean())
            rows.append(dict(task=task, regime=reg, n_base=len(base), n_bounds=len(bnd), base=base.TAR.mean(),
                             bounds=bnd.TAR.mean(), delta=np.nanmean(diff),
                             ci=1.96 * np.nanstd(diff, ddof=1) / np.sqrt(np.sum(~np.isnan(diff))),
                             post=np.mean(post), post_gain=np.mean(post) - np.mean(bpost),
                             within_gain=np.mean(within) - np.mean(bwithin)))
    out = pd.DataFrame(rows).round(4)
    out.to_csv(REPO / "results_clean" / "_summary" / "sampling_decoding.csv", index=False)
    print(out.to_string(index=False))
    lines = ["\\begin{tabular}{ll ccc cc}", "\\toprule",
             "Task & Decoding & No markers & Markers & Change & Post-marker & Within-segment \\\\", "\\midrule"]
    for task, _, _, lab in TASKS:
        for i, (reg, rlab) in enumerate(REGIMES):
            r = out[(out.task == task) & (out.regime == reg)].iloc[0]
            pp = ("---", "---") if task.startswith("random") else (f"${r.post_gain:+.2f}$", f"${r.within_gain:+.2f}$")
            lines.append(f"{'\\task{' + lab + '}' if i == 0 else ''} & {rlab} & {r.base:.2f} & {r.bounds:.2f} & "
                         f"${r.delta:+.2f} \\pm {r.ci:.2f}$ & {pp[0]} & {pp[1]} \\\\")
        if task != TASKS[-1][0]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "sampling.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/sampling.tex")


if __name__ == "__main__":
    main()
