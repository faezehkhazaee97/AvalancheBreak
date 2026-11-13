#!/usr/bin/env python3
"""Do markers change how errors cluster along the sequence? (Reviewer question.)

For every stored generation at N = 64 of the four stronger open models, the per-position error
indicator e_i (1 = wrong) is computed under the paper's index scoring and under the edit-distance
alignment of scripts/analyze_aligned_scoring.py. Three quantities are pooled over generations:

  rate   the error rate, mean of e_i
  p11    P(e_{i+1} = 1 | e_i = 1)   errors following an error (persistence)
  p01    P(e_{i+1} = 1 | e_i = 0)   errors following a correct position
  run    mean length of a maximal run of consecutive errors

grouped by marker density: no markers (k = 0), a marker after every character (k = 1), and sparse
markers (k in {2, 4, 8, 16}). Under index scoring a single dropped or inserted character makes
every later position wrong, so p11 near 1 there measures cascading misalignment; under aligned
scoring it measures genuine clustering of local errors. Formats: spaced letters and digits
(position-local tasks pooled, every alphabet) and the four dependency tasks pooled.

Outputs: results_clean/_summary/error_autocorrelation.csv, paper_tmlr/tables/autocorr.tex
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load  # noqa: E402
from analyze_aligned_scoring import clean, index_vector, aligned_vector  # noqa: E402

MODELS = [("Qwen2.5 14B", "Qwen2.5-14B-Instruct"), ("Qwen2.5 32B", "Qwen2.5-32B-Instruct"),
          ("Qwen3.8 27B", "Qwen3.8-27B"), ("Gemma 4 31B", "gemma-4-31B-it")]
FORMATS = [("Spaced letters", ["cyclic_shift_spaced", "random_permutation_spaced", "conditional_shift_spaced"]),
           ("Digits", ["cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"]),
           ("Dependency tasks", ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"])]
GROUPS = [("none", [0]), ("k1", [1]), ("sparse", [2, 4, 8, 16])]
N = 64


def text(r):
    g = r.get("Raw_Model_Output", "")
    g = "" if pd.isna(g) else str(g).strip().split("\n")[0]
    return clean(g) if g else clean(r.LLM_Output)


def stats(errs):
    """errs: list of 0/1 error vectors."""
    e = np.concatenate(errs)
    a = np.concatenate([v[:-1] for v in errs])
    b = np.concatenate([v[1:] for v in errs])
    runs = sum(int(((np.diff(np.concatenate([[0], v])) == 1)).sum()) for v in errs)
    return dict(n=len(errs), rate=e.mean(),
                p11=b[a == 1].mean() if (a == 1).any() else np.nan,
                p01=b[a == 0].mean() if (a == 0).any() else np.nan,
                run=e.sum() / runs if runs else np.nan)


def main():
    rows = []
    for label, tag in MODELS:
        for fmt, tasks in FORMATS:
            vecs = {(g, s): [] for g, _ in GROUPS for s in ("index", "aligned")}
            for task in tasks:
                d = load(task, tag)
                if d is None:
                    continue
                d = d[d.Seq_Length.astype(int) == N]
                for _, r in d.iterrows():
                    k = int(r.Interval)
                    g = next((g for g, ks in GROUPS if k in ks), None)
                    if g is None:
                        continue
                    t, p = clean(r.Ground_Truth), text(r)
                    vecs[(g, "index")].append(1 - index_vector(t, p))
                    vecs[(g, "aligned")].append(1 - aligned_vector(t, p))
            for (g, s), v in vecs.items():
                if v:
                    rows.append(dict(model=label, format=fmt, group=g, scorer=s, **stats(v)))
            print("done", label, fmt, flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(REPO / "results_clean" / "_summary" / "error_autocorrelation.csv", index=False)
    print(out.pivot_table(index=["model", "format", "scorer"], columns="group", values=["rate", "p11", "p01", "run"])
          .round(3).to_string())

    lines = [r"\begin{tabular}{ll ccc ccc ccc}", r"\toprule",
             r"& & \multicolumn{3}{c}{No markers} & \multicolumn{3}{c}{$k = 1$} & \multicolumn{3}{c}{$k \in \{2,4,8,16\}$} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
             r"Model & Format & rate & $p_{1|1}$ & $p_{1|0}$ & rate & $p_{1|1}$ & $p_{1|0}$ & rate & $p_{1|1}$ & $p_{1|0}$ \\",
             r"\midrule"]
    for scorer in ("index", "aligned"):
        lines.append(r"\multicolumn{11}{l}{\emph{" + ("Index scoring" if scorer == "index" else "Aligned scoring") + r"}} \\")
        for label, _ in MODELS:
            first = True
            for fmt, _t in FORMATS:
                cells = []
                for g, _k in GROUPS:
                    x = out[(out.model == label) & (out.format == fmt) & (out.group == g) & (out.scorer == scorer)]
                    cells += ["---"] * 3 if x.empty else [f"{x.rate.iloc[0]:.2f}", f"{x.p11.iloc[0]:.2f}", f"{x.p01.iloc[0]:.2f}"]
                lines.append(f"{label if first else ''} & {fmt} & " + " & ".join(cells) + r" \\")
                first = False
        if scorer == "index":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "autocorr.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/autocorr.tex")


if __name__ == "__main__":
    main()
