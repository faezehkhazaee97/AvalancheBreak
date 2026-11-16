#!/usr/bin/env python3
"""How much of the tokenization-independent gain survives edit-distance alignment?

scripts/analyze_aligned_scoring.py answers this for the dependency tasks. A reviewer asked the
same question of the position-local controls, where it separates two readings of Finding 2: do
markers help because they stop the model losing its place (a gain that alignment repairs away),
or because each character is transformed more accurately (a gain that survives alignment)?

For each control task and model at N = 64, the marker and no-marker runs are scored twice, index
by index and under the minimum-edit-distance alignment of analyze_aligned_scoring.py, and we
report the gain under each. Delta_aligned / Delta_index is the share of the gain that is not
alignment repair; 1 minus that is the share that is.

Raw generations are stored for every spaced-letter and digit control run (Appendix B.1), so no
cell is reconstructed from clipped output.

Output: results_clean/_summary/aligned_controls.csv
        paper_tmlr/tables/aligned_controls.tex
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS, KS  # noqa: E402
from analyze_aligned_scoring import clean, index_vector, aligned_vector  # noqa: E402

BASE = ["cyclic_shift", "random_permutation", "conditional_shift"]
FORMATS = [("_spaced", "Spaced letters"), ("_digits", "Digits")]
STRONG = ["Qwen2.5 14B", "Qwen2.5 32B", "Qwen3.8 27B", "Gemma 4 31B"]
N = 64


def mean_acc(df, scorer):
    """Mean per-character accuracy over the generations of a cell, under one scorer."""
    out = []
    for _, r in df.iterrows():
        t, p = clean(r.Ground_Truth), clean(r.LLM_Output)
        if not t:
            continue
        out.append(scorer(t, p).mean())
    return float(np.mean(out)) if out else float("nan")


def main():
    rows = []
    for label, tag in MODELS:
        if label not in STRONG:
            continue
        for b in BASE:
            for suf, fmt in FORMATS:
                d = load(b + suf, tag)
                if d is None:
                    continue
                d = d[d.Seq_Length.astype(int) == N]
                base = d[d.Method == "Base"]
                bnd = d[(d.Method == "Cellular_Bounds") & d.Interval.astype(int).isin(KS)]
                if len(base) < 2 or len(bnd) < 2:
                    continue
                rec = dict(model=label, task=b, fmt=fmt, n_base=len(base), n_marked=len(bnd))
                for name, scorer in (("index", index_vector), ("aligned", aligned_vector)):
                    a0, a1 = mean_acc(base, scorer), mean_acc(bnd, scorer)
                    rec[f"base_{name}"], rec[f"marked_{name}"] = a0, a1
                    rec[f"delta_{name}"] = a1 - a0
                rows.append(rec)
    if not rows:
        print("no control cells found")
        return
    o = pd.DataFrame(rows)
    o["survives"] = o.delta_aligned / o.delta_index.replace(0, np.nan)
    o = o.round(4)
    out = REPO / "results_clean" / "_summary" / "aligned_controls.csv"
    o.to_csv(out, index=False)
    print(f"wrote {out} ({len(o)} cells)")
    pd.set_option("display.width", 200)
    print(o[["model", "task", "fmt", "delta_index", "delta_aligned", "survives"]].to_string(index=False))
    s = o.survives.dropna()
    print("\nshare of the gain surviving alignment: median %.0f%%, range %.0f%% to %.0f%%"
          % (100 * s.median(), 100 * s.min(), 100 * s.max()))

    lines = ["\\begin{tabular}{ll cc cc c}", "\\toprule",
             "& & \\multicolumn{2}{c}{Index scoring} & \\multicolumn{2}{c}{Aligned} & \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
             "Model & Format & No markers & $\\Delta$ & No markers & $\\Delta$ & Survives \\\\",
             "\\midrule"]
    for label in STRONG:
        first = True
        for _, fmt in FORMATS:
            x = o[(o.model == label) & (o.fmt == fmt)]
            if x.empty:
                continue
            # mean over the three position-local tasks, as elsewhere in the paper
            bi, di = x.base_index.mean(), x.delta_index.mean()
            ba, da = x.base_aligned.mean(), x.delta_aligned.mean()
            lines.append(f"{label if first else ''} & {fmt} & {bi:.2f} & ${di:+.2f}$ & "
                         f"{ba:.2f} & ${da:+.2f}$ & {100 * da / di:.0f}\\% \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    t = REPO / "paper_tmlr" / "tables" / "aligned_controls.tex"
    t.write_text("\n".join(lines) + "\n")
    print(f"wrote {t}")


if __name__ == "__main__":
    main()
