#!/usr/bin/env python3
"""LaTeX tables for head ablation v2 (results/ablation_v2/<model>.csv).

For each format (merged letters, spaced letters, digits) and length, pooled over k in {2, 8}:
token accuracy without markers, with markers, with the top-10 marker-attending heads blocked
from markers, with 10 random heads blocked (control), and with the top heads blocked from
as many random non-marker tokens (control); plus the share of the marker gain each removes
and the paired drop (markers minus intervention) with a 95% interval over generations.

Writes paper_tmlr/tables/ablation_<tag>.tex.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FORMATS = [("random_permutation", "Letters"), ("random_permutation_spaced", "Spaced letters"),
           ("random_permutation_digits", "Digits")]


def table(tag):
    a = pd.read_csv(REPO / "results" / "ablation_v2" / f"{tag}.csv")
    lines = [r"\begin{tabular}{llccccccc}", r"\toprule",
             r"& & & & \multicolumn{3}{c}{Markers, with heads blocked} & \multicolumn{2}{c}{Share of gain removed} \\",
             r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
             r"Format & $N$ & No markers & Markers & Top 10 & Random 10 & Top 10, sham & Top 10 & Controls \\",
             r"\midrule"]
    for task, label in FORMATS:
        for N in sorted(a.Seq_Length.unique()):
            s = a[(a.Task == task) & (a.Seq_Length == N)]
            m = s.groupby("Condition").TAR.mean()
            gain = m["bounds"] - m["base"]
            share = lambda c: (m["bounds"] - m[c]) / gain if gain > 0.03 else np.nan
            ctrl = max(0.0, np.nanmax([share("ctrl_random"), share("ctrl_sham")])) if gain > 0.03 else np.nan
            fmt = lambda x: "--" if np.isnan(x) else f"{100 * x:.0f}\\%"
            cells = [label, str(N)] + [f"{m[c]:.2f}" for c in ("base", "bounds", "ablate_top", "ctrl_random", "ctrl_sham")]
            if np.isnan(ctrl):
                cells += ["--", "--"]
            else:
                cells += [fmt(share("ablate_top")), f"$\\le$ {fmt(ctrl)}" if ctrl >= 0.005 else "0\\%"]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = REPO / "paper_tmlr" / "tables" / f"ablation_{tag.split('-')[1].lower()}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    p = a[a.Condition != "base"].pivot_table(index=["Task", "Seq_Length", "Interval", "Trial"],
                                              columns="Condition", values="TAR")
    for c in ["ablate_top", "ctrl_random", "ctrl_sham"]:
        d = (p.bounds - p[c]).groupby(level=0).agg(["mean", "sem"])
        print(f"{tag} paired drop, markers - {c}: " +
              ", ".join(f"{t}: {r['mean']:+.3f} +/- {1.96 * r['sem']:.3f}" for t, r in d.iterrows()))
    print(f"Wrote {out}")


if __name__ == "__main__":
    for tag in sys.argv[1:] or ["Qwen2.5-32B-Instruct", "Qwen2.5-7B-Instruct"]:
        table(tag)
