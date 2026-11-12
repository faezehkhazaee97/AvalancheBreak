#!/usr/bin/env python3
"""Did the model follow the whole instruction, not just transform the characters?

Character accuracy strips markers and spaces from both sides, so a model can score well while
ignoring the requested output format. This script adds two stricter measures over every stored
generation of the seven open models (N <= 64):

  exact      the cleaned prediction equals the cleaned ground truth (the stored SAR)
  compliant  the raw output's first line carries markers in exactly the right places: the same
             number as the expected output and at the same character offsets (spaces ignored)

For no-marker runs, compliance means the output carries no markers at all. Generations whose
raw text was not stored are skipped and counted.

Output: results_clean/_summary/compliance.csv, paper_tmlr/tables/compliance.tex
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS, BLOCKS  # noqa: E402

LENGTHS = [4, 8, 16, 32, 64]


def marker_positions(text):
    """Offsets of '|' among the non-space characters of the first line."""
    line = str(text).strip().split("\n")[0].replace(" ", "")
    return [i for i, c in enumerate(line) if c == "|"]


def expected_positions(n, k):
    """Where insert_boundaries puts markers in an n-character answer at spacing k."""
    if k <= 0:
        return []
    out, seen = [], 0
    for i in range(n):
        seen += 1
        if seen % k == 0 and i != n - 1:
            out.append(i + 1 + len(out))
    return out


def main():
    rows, skipped, total = [], 0, 0
    for label, tag in MODELS:
        if tag is None:
            continue
        for block, tasks in BLOCKS:
            if "reset" in block:
                continue
            for task in tasks:
                d = load(task, tag)
                if d is None or "Raw_Model_Output" not in d.columns:
                    continue
                d = d.assign(N=d.Seq_Length.astype(int), k=d.Interval.astype(int))
                d = d[d.N.isin(LENGTHS) & (d.k != 3)]
                total += len(d)
                d = d[d.Raw_Model_Output.map(lambda x: isinstance(x, str) and x != "")]
                skipped += 0
                for cls, g in (("no markers", d[d.k == 0]), ("markers", d[d.k > 0])):
                    if g.empty:
                        continue
                    ok = [marker_positions(r.Raw_Model_Output) == expected_positions(len(r.Ground_Truth), r.k)
                          for r in g.itertuples()]
                    rows.append(dict(model=label, block=block, task=task, cls=cls, n=len(g),
                                     exact=g.SAR.astype(float).mean(), compliant=sum(ok) / len(ok)))
        print("done", label, flush=True)
    skipped = total - sum(r["n"] for r in rows)
    print(f"generations with no stored raw output (skipped): {skipped} of {total}")
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results_clean" / "_summary" / "compliance.csv", index=False)
    w = out.assign(ex=out.exact * out.n, co=out.compliant * out.n).groupby(["block", "cls"])[["n", "ex", "co"]].sum()
    w["exact"], w["compliant"] = w.ex / w.n, w.co / w.n
    print(w[["n", "exact", "compliant"]].round(3).to_string())
    m = out.assign(ex=out.exact * out.n, co=out.compliant * out.n).groupby(["model", "cls"])[["n", "ex", "co"]].sum()
    m["exact"], m["compliant"] = m.ex / m.n, m.co / m.n
    print(m[["n", "exact", "compliant"]].round(3).to_string())

    lines = ["\\begin{tabular}{ll cc cc}", "\\toprule",
             "& & \\multicolumn{2}{c}{No markers} & \\multicolumn{2}{c}{Markers} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
             "Model & Task group & Exact & Compliant & Exact & Compliant \\\\", "\\midrule"]
    g = out.assign(ex=out.exact * out.n, co=out.compliant * out.n)
    for label, _ in MODELS:
        first = True
        for block, _t in BLOCKS:
            if "reset" in block:
                continue
            x = g[(g.model == label) & (g.block == block)]
            if x.empty:
                continue
            cells = []
            for cls in ("no markers", "markers"):
                y = x[x.cls == cls]
                cells += ["---", "---"] if y.empty else [f"{y.ex.sum() / y.n.sum():.2f}",
                                                         f"{y.co.sum() / y.n.sum():.2f}"]
            lines.append(f"{label if first else ''} & {block} & " + " & ".join(cells) + " \\\\")
            first = False
        if not first:
            lines.append("\\midrule")
    # pooled over models, weighted by the number of generations (the averages the text quotes)
    first = True
    for block, _t in BLOCKS:
        if "reset" in block:
            continue
        x = g[g.block == block]
        cells = []
        for cls in ("no markers", "markers"):
            y = x[x.cls == cls]
            cells += ["---", "---"] if y.empty else [f"{y.ex.sum() / y.n.sum():.2f}", f"{y.co.sum() / y.n.sum():.2f}"]
        lines.append(f"{'All models, weighted' if first else ''} & {block} & " + " & ".join(cells) + " \\\\")
        first = False
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    (REPO / "paper_tmlr" / "tables" / "compliance.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/compliance.tex")


if __name__ == "__main__":
    main()
