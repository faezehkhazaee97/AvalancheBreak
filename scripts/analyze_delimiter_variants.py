#!/usr/bin/env python3
"""Analysis of scripts/run_delimiter_variants.py: does the segmentation gain need a delimiter
the model has to copy, or is an out-of-band separator enough?

  base        no delimiter
  bar_k4      the paper's marker every 4 characters, copied through (reference)
  space_both  a space every 4 characters, kept in the answer
  space_in    a space every 4 characters in the input, answer written unbroken

space_in is the condition that matters: the input is segmented but the answer carries no
separator, so nothing has to be reproduced. If it reaches bar_k4 the gain is about the input
being grouped; if it sits at base the gain needs the output segmented too, which is what the
one-sided ablation of Section 8.2 implies.

Output: results_clean/_summary/delimiter_variants.csv
        paper_tmlr/tables/delimiter_variants.tex
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODELS = [("Qwen2.5 32B", "Qwen2.5-32B-Instruct"), ("Gemma 4 31B", "gemma-4-31B-it")]
CONDN = {"base": "No delimiter", "bar_k4": "Bar, both sides",
         "space_both": "Space, both sides", "space_in": "Space, input only"}
TASKN = {"running_xor": "Running XOR", "running_sum_local": "Running sum (mod 4)",
         "random_permutation_digits": "Random permutation"}


def main():
    rows = []
    for label, tag in MODELS:
        f = REPO / "results" / "delimiter_variants" / f"{tag}.csv"
        if not f.exists():
            print(f"no results for {label}")
            continue
        d = pd.read_csv(f)
        for (task, N), g in d.groupby(["Task", "Seq_Length"]):
            base = g[g.Condition == "base"]
            if base.empty:
                continue
            b = base.TAR.mean()
            for cond in CONDN:
                x = g[g.Condition == cond]
                if x.empty:
                    continue
                rows.append(dict(model=label, task=task, N=N, cond=cond, n=len(x),
                                 acc=x.TAR.mean(), se=x.TAR.std(ddof=1) / np.sqrt(len(x)),
                                 delta=x.TAR.mean() - b))
    if not rows:
        print("nothing to analyse yet")
        return
    o = pd.DataFrame(rows).round(4)
    out = REPO / "results_clean" / "_summary" / "delimiter_variants.csv"
    o.to_csv(out, index=False)
    print(f"wrote {out} ({len(o)} rows)")
    pd.set_option("display.width", 200)
    print(o.pivot_table(index=["model", "task", "N"], columns="cond", values="delta").round(3).to_string())

    lines = ["\\begin{tabular}{ll cccc}", "\\toprule",
             "& & \\multicolumn{4}{c}{Accuracy ($\\Delta$ against no delimiter)} \\\\",
             "\\cmidrule(lr){3-6}",
             "Model & Task & " + " & ".join(CONDN[c] for c in CONDN) + " \\\\", "\\midrule"]
    for label, _ in MODELS:
        first = True
        for task in TASKN:
            x = o[(o.model == label) & (o.task == task) & (o.N == 64)]
            if x.empty:
                continue
            cells = []
            for cond in CONDN:
                r = x[x.cond == cond]
                if r.empty:
                    cells.append("--")
                elif cond == "base":
                    cells.append(f"{r.acc.iloc[0]:.2f}")
                else:
                    cells.append(f"{r.acc.iloc[0]:.2f} (${r.delta.iloc[0]:+.2f}$)")
            lines.append(f"{label if first else ''} & {TASKN[task]} & " + " & ".join(cells) + " \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    t = REPO / "paper_tmlr" / "tables" / "delimiter_variants.tex"
    t.write_text("\n".join(lines) + "\n")
    print(f"wrote {t}")


if __name__ == "__main__":
    main()
