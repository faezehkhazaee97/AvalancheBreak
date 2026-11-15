#!/usr/bin/env python3
"""Every pooled cell of the two tokenization controls (spaced letters, digits), task by model.

Per cell (N <= 64, k in {1, 2, 4, 8, 16}, all alphabet sizes, k = 3 excluded): the length-matched
no-marker accuracy, the marker accuracy, Delta, the share of remaining errors removed
Delta / (1 - Base_matched), and three z values:
  z          the paper's analytic z, SE(Bounds) = s / sqrt(n) over all marker generations;
  z_strat    a fixed-grid stratified z: the (N, k, A) cells are fixed design strata, so
             SE(Bounds)^2 = sum_c w_c^2 s_c^2 / n_c with w_c the share of marker generations in
             cell c, and SE(Base at N)^2 likewise over the alphabet sizes within N;
  z_cluster  the cell-level bootstrap of Appendix C (statistics_robustness.csv), open models only.
Also printed: per model and format, the mean over the two strictly isomorphic tasks (cyclic_shift,
random_permutation) next to conditional_shift, whose digit version is easier by construction.
Outputs: results_clean/_summary/control_cells.csv, paper_tmlr/tables/control_cells.tex
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS  # noqa: E402

LENGTHS = [4, 8, 16, 32, 64]
SIG_Z = 3.54
TASKS = ["cyclic_shift", "random_permutation", "conditional_shift"]
FORMATS = [("_spaced", "Spaced"), ("_digits", "Digits")]


def strat(d):
    d = d[d.Interval.astype(int) != 3].copy()
    d["N"] = d.Seq_Length.astype(int); d["k"] = d.Interval.astype(int); d["TAR"] = d.TAR.astype(float)
    a = next((c for c in ("Alphabet_Size", "Modulus", "Axis_Val") if c in d.columns), None)
    d["A"] = d[a].astype(str) if a else "0"
    d = d[d.N.isin(LENGTHS)]
    b, c = d[d.k == 0], d[d.k > 0]
    p = c.groupby("N").size() / len(c)
    base_N = b.groupby("N").TAR.mean().reindex(p.index)
    base_m = float((p * base_N).sum()); bounds_m = float(c.TAR.mean()); delta = bounds_m - base_m
    se_b_N = b.groupby("N").TAR.sem().reindex(p.index).fillna(0)
    se = np.sqrt(c.TAR.sem() ** 2 + (p ** 2 * se_b_N ** 2).sum())
    # stratified: marker side over (N, k, A) cells, baseline side over A within N
    g = c.groupby(["N", "k", "A"]).TAR.agg(["var", "count"])
    w = g["count"] / g["count"].sum()
    var_c = (w ** 2 * g["var"].fillna(0) / g["count"]).sum()
    var_b = 0.0
    for N, pN in p.items():
        gb = b[b.N == N].groupby("A").TAR.agg(["var", "count"])
        wb = gb["count"] / gb["count"].sum()
        var_b += pN ** 2 * (wb ** 2 * gb["var"].fillna(0) / gb["count"]).sum()
    se_s = np.sqrt(var_c + var_b)
    c64, b64 = c[c.N == 64], b[b.N == 64]
    d64 = c64.TAR.mean() - b64.TAR.mean() if len(c64) and len(b64) else np.nan
    return dict(base=base_m, bounds=bounds_m, delta=delta, z=delta / se, z_strat=delta / se_s,
                share=delta / (1 - base_m) if base_m < 1 else np.nan, delta64=d64)


def main():
    rob = pd.read_csv(REPO / "results_clean" / "_summary" / "statistics_robustness.csv")
    rows = []
    for suf, fmt in FORMATS:
        for task in TASKS:
            for label, tag in MODELS:
                if tag is None:
                    continue
                d = load(task + suf, tag)
                if d is None:
                    continue
                r = strat(d)
                m = rob[(rob.task == task + suf) & (rob.model == label)]
                r.update(format=fmt, task=task, model=label, z_cluster=float(m.cluster_z.iloc[0]) if len(m) else np.nan)
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(REPO / "results_clean" / "_summary" / "control_cells.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    strong = ["Qwen2.5 14B", "Qwen2.5 32B", "Qwen3.8 27B", "Gemma 4 31B"]
    for name, sub in [("strong", t[t.model.isin(strong)]), ("small", t[t.model.isin(["Qwen2.5 7B", "Mistral 7B", "Phi-3.5 mini"])])]:
        print(f"{name}: {len(sub)} cells; |z|>=3.54 analytic {(sub.z.abs() >= SIG_Z).sum()}, stratified {(sub.z_strat.abs() >= SIG_Z).sum()}, "
              f"cluster {(sub.z_cluster.abs() >= SIG_Z).sum()}; non-significant analytic: "
              + "; ".join(f"{r.model} {r.format} {r.task} ({r.z:.2f})" for r in sub.itertuples() if abs(r.z) < SIG_Z))
    print("\nmodel means: base, delta, share over all three tasks vs the two isomorphic tasks; conditional_shift alone")
    for (fmt, model), g in t.groupby(["format", "model"], sort=False):
        iso = g[g.task != "conditional_shift"]; cs = g[g.task == "conditional_shift"]
        sh = lambda x: x.delta.mean() / (1 - x.base.mean())
        print(f"  {fmt:7s} {model:16s} all: base {g.base.mean():.3f} delta {g.delta.mean():+.3f} share {sh(g):.0%} | "
              f"isomorphic: base {iso.base.mean():.3f} delta {iso.delta.mean():+.3f} share {sh(iso):.0%} | cond: base {cs.base.mean():.3f} delta {cs.delta.mean():+.3f}")
    names = {"cyclic_shift": "\\task{cyclic\\_shift}", "random_permutation": "\\task{random\\_permutation}",
             "conditional_shift": "\\task{conditional\\_shift}"}
    f = lambda v: "--" if pd.isna(v) else f"{v:.1f}"
    lines = ["\\begin{tabular}{lll ccc c ccc}", "\\toprule",
             "Model & Format & Task & No markers & Markers & $\\Delta$ & Share removed & $z$ & $z$ (stratified) & $z$ (cell bootstrap) \\\\", "\\midrule"]
    for model in t.model.unique():
        g = t[t.model == model]; first = True
        for r in g.itertuples():
            lines.append(f"{model if first else ''} & {r.format} & {names[r.task]} & {r.base:.2f} & {r.bounds:.2f} & ${r.delta:+.2f}$ & "
                         f"{100 * r.share:.0f}\\% & {r.z:.1f} & {r.z_strat:.1f} & {f(r.z_cluster)} \\\\".replace("$-0.00$", "$0.00$").replace("$+0.00$", "$0.00$"))
            first = False
        lines.append("\\addlinespace")
    lines[-1:] = ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "control_cells.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/control_cells.tex")


if __name__ == "__main__":
    main()
