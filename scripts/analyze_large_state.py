#!/usr/bin/env python3
"""Analysis of scripts/run_large_state.py: does the blind spot survive a larger state space?

running_sum_local at modulus 8, 16 and 32 (Qwen2.5-32B, spaced single-character symbols,
N in {32, 64}, k in {4, 16} pooled, 20 trials). Reports accuracy without and with markers, and
post-marker and within-segment accuracy against the no-marker run at the same positions.
Chance is 1/M, so the post-marker column is read against that, not against zero.

Outputs: results_clean/_summary/large_state.csv, paper_tmlr/tables/large_state.tex
"""
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}


def vec(truth, pred):
    pred = "" if pd.isna(pred) else str(pred)
    return np.array([i < len(pred) and pred[i] == c for i, c in enumerate(truth)], dtype=float)


def main():
    d = pd.read_csv(REPO / "results" / "large_state" / "Qwen2.5-32B-Instruct.csv", dtype=TEXT, keep_default_na=False)
    d = d.assign(M=d.Modulus.astype(int), N=d.Seq_Length.astype(int), k=d.Interval.astype(int),
                 TAR=d.TAR.astype(float))
    rows = []
    for M in sorted(d.M.unique()):
        x = d[d.M == M]
        base, bnd = x[x.k == 0], x[x.k > 0]
        ref = {N: np.mean([vec(r.Ground_Truth, r.LLM_Output) for r in base[base.N == N].itertuples()], axis=0)
               for N in sorted(x.N.unique())}
        post, within, bpost, bwithin = [], [], [], []
        for r in bnd.itertuples():
            v = vec(r.Ground_Truth, r.LLM_Output)
            idx = np.arange(len(v)); p = (idx > 0) & (idx % r.k == 0)
            post.append(v[p].mean()); within.append(v[~p].mean())
            bpost.append(ref[r.N][p].mean()); bwithin.append(ref[r.N][~p].mean())
        diff = bnd.TAR.mean() - base.TAR.mean()
        se = np.sqrt(base.TAR.var(ddof=1) / len(base) + bnd.TAR.var(ddof=1) / len(bnd))
        rows.append(dict(M=M, chance=round(1 / M, 3), base=base.TAR.mean(), bounds=bnd.TAR.mean(), delta=diff,
                         ci=1.96 * se, post=np.mean(post), post_gain=np.mean(post) - np.mean(bpost),
                         within_gain=np.mean(within) - np.mean(bwithin), n=len(x)))
    o = pd.DataFrame(rows).round(4)
    o.to_csv(REPO / "results_clean" / "_summary" / "large_state.csv", index=False)
    print(o.to_string(index=False))
    lines = ["\\begin{tabular}{cc cc c cc}", "\\toprule",
             "$M$ & Chance & No markers & Markers & Change & Post-marker & Within-segment \\\\", "\\midrule"]
    for r in o.itertuples():
        lines.append(f"{r.M} & {r.chance:.3f} & {r.base:.2f} & {r.bounds:.2f} & ${r.delta:+.2f} \\pm {r.ci:.2f}$ & "
                     f"{r.post:.2f} (${r.post_gain:+.2f}$) & ${r.within_gain:+.2f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "large_state.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/large_state.tex")


if __name__ == "__main__":
    main()
