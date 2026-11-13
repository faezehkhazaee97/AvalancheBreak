#!/usr/bin/env python3
"""Analysis of scripts/run_paired_inputs.py: how much does pairing the inputs tighten the marker
effect? For every (task, alphabet or modulus, N, k > 0) cell of a paired run:

  delta_paired     mean over trials of TAR(k) - TAR(0) on the same input
  se_paired        standard error of that mean of differences
  se_unpaired      standard error of the difference of the two means, as if the runs were unpaired
                   (the design of the main grid)
  r                correlation of TAR(k) and TAR(0) across trials
  delta_main       the main grid's estimate for the same cell (different inputs per condition)

and a reproduction check: the paired run's no-marker rows reuse the main grid's seeds, so with
greedy decoding they should match the main grid's Base rows exactly.

Outputs: results_clean/_summary/paired_inputs.csv, paper_tmlr/tables/paired.tex
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load  # noqa: E402

TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}
MODELS = [("Qwen2.5 7B", "Qwen2.5-7B-Instruct"), ("Mistral 7B", "Mistral-7B-Instruct-v0.3"),
          ("Phi-3.5 mini", "Phi-3.5-mini-instruct"), ("Qwen2.5 32B", "Qwen2.5-32B-Instruct")]
FAM = {"Spaced letters": ["cyclic_shift_spaced", "random_permutation_spaced", "conditional_shift_spaced"],
       "Digits": ["cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"],
       "Dependency": ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]}


def main():
    cells, repro = [], []
    for label, tag in MODELS:
        f = REPO / "results" / "paired_inputs" / f"{tag}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f, dtype=TEXT, keep_default_na=False)
        p = p.assign(N=p.Seq_Length.astype(int), k=p.Interval.astype(int), axis=p.Axis.astype(int),
                     t=p.Trial.astype(int), TAR=p.TAR.astype(float))
        for fam, tasks in FAM.items():
            for task in tasks:
                x = p[p.Task == task]
                if x.empty:
                    continue
                g = load(task, tag)
                ac = next((c for c in ("Alphabet_Size", "Modulus") if c in g.columns), None) if g is not None else None
                if g is not None:
                    g = g.assign(N=g.Seq_Length.astype(int), k=g.Interval.astype(int), t=g.Trial.astype(int),
                                 TAR=g.TAR.astype(float), axis=g[ac].astype(int) if ac else 2)
                    g = g.drop_duplicates(["N", "k", "axis", "t"])
                    # reproduction check of the no-marker rows
                    b0 = x[x.k == 0].set_index(["N", "axis", "t"]).LLM_Output
                    g0 = g[g.k == 0].set_index(["N", "axis", "t"]).LLM_Output
                    common = b0.index.intersection(g0.index)
                    if len(common):
                        repro.append(dict(model=label, task=task, n=len(common),
                                          same=float((b0.loc[common] == g0.loc[common]).mean())))
                for (N, ax), grp in x.groupby(["N", "axis"]):
                    base = grp[grp.k == 0].set_index("t").TAR
                    for k, gk in grp[grp.k > 0].groupby("k"):
                        pair = gk.set_index("t").TAR.to_frame("m").join(base.rename("b"), how="inner")
                        if len(pair) < 5:
                            continue
                        d = pair.m - pair.b
                        main = np.nan
                        if g is not None:
                            gm = g[(g.N == N) & (g.axis == ax) & (g.k == k)].TAR
                            gb = g[(g.N == N) & (g.axis == ax) & (g.k == 0)].TAR
                            if len(gm) and len(gb):
                                main = gm.mean() - gb.mean()
                        cells.append(dict(model=label, family=fam, task=task, N=N, axis=ax, k=k, n=len(pair),
                                          delta_paired=d.mean(), se_paired=d.std(ddof=1) / np.sqrt(len(d)),
                                          se_unpaired=np.sqrt(pair.m.var(ddof=1) / len(d) + pair.b.var(ddof=1) / len(d)),
                                          r=pair.m.corr(pair.b) if pair.m.std() > 0 and pair.b.std() > 0 else np.nan,
                                          delta_main=main))
    o = pd.DataFrame(cells).round(4)
    o.to_csv(REPO / "results_clean" / "_summary" / "paired_inputs.csv", index=False)
    rp = pd.DataFrame(repro)
    if len(rp):
        print("no-marker rows identical to the main grid (share, by model):")
        print(rp.groupby("model").apply(lambda z: np.average(z.same, weights=z.n)).round(3).to_string())
    s = o.groupby(["model", "family"]).agg(cells=("k", "size"), delta_paired=("delta_paired", "mean"),
                                           delta_main=("delta_main", "mean"), se_paired=("se_paired", "mean"),
                                           se_unpaired=("se_unpaired", "mean"), r=("r", "mean"))
    s["ratio"] = s.se_paired / s.se_unpaired
    print(s.round(3).to_string())

    lines = ["\\begin{tabular}{ll c cc cc c}", "\\toprule",
             "& & & \\multicolumn{2}{c}{$\\Delta$} & \\multicolumn{2}{c}{Cell SE} & \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Model & Format & Cells & Main grid & Paired & Unpaired & Paired & $r$ \\\\", "\\midrule"]
    for label, _ in MODELS:
        first = True
        for fam in FAM:
            if (label, fam) not in s.index:
                continue
            r = s.loc[(label, fam)]
            lines.append(f"{label if first else ''} & {fam} & {int(r.cells)} & ${r.delta_main:+.3f}$ & ${r.delta_paired:+.3f}$ & "
                         f"{r.se_unpaired:.3f} & {r.se_paired:.3f} & {r.r:.2f} \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "paired.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/paired.tex")


if __name__ == "__main__":
    main()
