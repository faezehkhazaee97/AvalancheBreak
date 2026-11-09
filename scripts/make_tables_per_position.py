#!/usr/bin/env python3
"""Write the paper's per-position appendix tables as LaTeX, straight from the analysis CSVs.

  paper_tmlr/tables/perpos.tex    : block means per model (Table tab:perpos)
  paper_tmlr/tables/distance.tex  : accuracy by distance after the marker, k = 16 (tab:distance)

Run scripts/analyze_per_position.py first.
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results_clean" / "_summary"
OUT = REPO / "paper_tmlr" / "tables"
MODELS = ["Qwen2.5 7B", "Qwen2.5 14B", "Qwen2.5 32B", "Mistral 7B", "Phi-3.5 mini",
          "Qwen3.8 27B", "Gemma 4 31B", "Gemini 3.5 Flash"]
BLOCKS = [("Position-local, letters", "Position-local tasks, letters"),
          ("Position-local, digits", "Position-local tasks, digits"),
          ("Dependency, as is", "Dependency tasks"),
          ("Dependency severed (reset)", "Dependency tasks, reset variants")]


def f(x):
    return f"{x:.2f}"


def signed(x):
    x = round(x, 2) + 0.0  # avoid "-0.00"
    return "$0.00$" if x == 0 else f"${'+' if x > 0 else '-'}{abs(x):.2f}$"


def perpos():
    v = pd.read_csv(SUMMARY / "per_position_values.csv")
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{Within a segment} & \multicolumn{3}{c}{Right after a marker} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Model & No markers & Markers & $\Delta$ & No markers & Markers & $\Delta$ \\", r"\midrule"]
    for block, title in BLOCKS:
        s = v[v.block == block]
        if s.empty:
            continue
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}} \\")
        for m in MODELS:
            x = s[s.model == m]
            if x.empty:
                continue
            bw, w, bp, p = x.base_at_within.mean(), x.within.mean(), x.base_at_post.mean(), x.post.mean()
            lines.append(f"{m} & {f(bw)} & {f(w)} & {signed(w - bw)} & {f(bp)} & {f(p)} & {signed(p - bp)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "perpos.tex").write_text("\n".join(lines) + "\n")


def distance():
    d = pd.read_csv(SUMMARY / "per_position_distance.csv")
    d = d[d.k == 16]
    bins = [("0", [0]), ("1", [1]), ("2--3", [2, 3]), ("4--7", [4, 5, 6, 7]), ("8--15", list(range(8, 16)))]
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"& & \multicolumn{5}{c}{Positions after the marker} \\", r"\cmidrule(lr){3-7}",
             "Tasks & Model & " + " & ".join(b for b, _ in bins) + r" \\", r"\midrule"]
    for block, title in [("Position-local, letters", "Position-local, letters"),
                         ("Position-local, digits", "Position-local, digits"),
                         ("Dependency, as is", "Dependency")]:
        s = d[d.block == block]
        if s.empty:
            continue
        first = True
        for m in MODELS:
            x = s[s.model == m]
            if x.empty:
                continue
            cells = " & ".join(f(x[x.d.isin(dd)].acc.mean()) for _, dd in bins)
            lines.append(f"{title if first else ''} & {m} & {cells} \\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (OUT / "distance.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    perpos()
    distance()
    print(f"Wrote {OUT / 'perpos.tex'} and {OUT / 'distance.tex'}")
