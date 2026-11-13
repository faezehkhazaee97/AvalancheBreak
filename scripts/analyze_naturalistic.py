#!/usr/bin/env python3
"""Naturalistic dependency tasks (reviewer question: is the blind spot an artefact of the synthetic
tasks?). tasks/bracket_depth (nesting depth mod 4 of a bracket string) and tasks/running_checksum
(digit-wise running checksum) carry the same one-step dependency as the four synthetic dependency
tasks but have a natural reading. For every model with a grid run: no-marker accuracy, the pooled
length-matched marker effect (every k > 0 against k = 0 at the same length and axis value, k = 3
excluded as in the main grid), and the per-position comparison of scripts/analyze_per_position.py
(post-marker and within-segment positions against the no-marker run at the same indices; k in
{2,4,8,16}; index scoring). The four synthetic dependency tasks are summarised the same way for
the same models as the reference row.

Outputs: results_clean/_summary/naturalistic.csv, paper_tmlr/tables/naturalistic.tex
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS as ALL_MODELS, LENGTHS  # noqa: E402
from analyze_aligned_scoring import summarize, index_vector  # noqa: E402

NAT = [("bracket_depth", "Bracket depth"), ("running_checksum", "Running checksum")]
SYN = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]


def pooled_delta(d):
    """Length-matched pooled marker effect in TAR: mean over marked rows of TAR minus the no-marker
    mean at the same length and axis value; k = 3 excluded; N <= 64."""
    d = d[d.Seq_Length.astype(int).isin(LENGTHS)].copy()
    d["N"] = d.Seq_Length.astype(int); d["k"] = d.Interval.astype(int); d["TAR"] = d.TAR.astype(float)
    axis = next((c for c in ("Alphabet_Size", "Modulus") if c in d.columns), None)
    keys = ["N"] + ([axis] if axis else [])
    base = d[d.k == 0].groupby(keys).TAR.mean()
    m = d[(d.k > 0) & (d.k != 3)]
    ref = m.set_index(keys).index.map(base)
    return float((m.TAR.values - np.asarray(ref, dtype=float)).mean())


def row(task, label, d):
    s = summarize(d, index_vector)
    return dict(task=task, model=label, n=len(d), base=s["base"], delta=pooled_delta(d),
                post=s["post"], post_gain=s["post_gain"], within_gain=s["within_gain"])


def main():
    rows = []
    for label, tag in ALL_MODELS:
        if tag is None or str(tag).startswith("gemini"):
            continue
        nat = [(t, load(t, tag)) for t, _ in NAT]
        if all(d is None for _, d in nat):
            continue
        for t, d in nat:
            if d is not None:
                rows.append(row(t, label, d))
        syn = [row(t, label, load(t, tag)) for t in SYN if load(t, tag) is not None]
        if syn:
            m = pd.DataFrame(syn)
            rows.append(dict(task="synthetic_mean", model=label, n=int(m.n.sum()),
                             **{c: float(m[c].mean()) for c in ("base", "delta", "post", "post_gain", "within_gain")}))
    out = pd.DataFrame(rows).round(4)
    if out.empty:
        print("no results yet"); return
    out.to_csv(REPO / "results_clean" / "_summary" / "naturalistic.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))

    names = dict(NAT + [("synthetic_mean", "Four synthetic tasks (mean)")])
    lines = ["\\begin{tabular}{ll cccc}", "\\toprule",
             "Task & Model & No-marker acc. & $\\Delta$ & Post-marker & Within-segment \\\\", "\\midrule"]
    for task in [t for t, _ in NAT] + ["synthetic_mean"]:
        first = True
        for label, _ in ALL_MODELS:
            x = out[(out.task == task) & (out.model == label)]
            if x.empty:
                continue
            r = x.iloc[0]
            tlab = names[task] if first else ""
            lines.append(f"{tlab} & {label} & {r.base:.2f} & ${r.delta:+.2f}$ & ${r.post_gain:+.2f}$ & ${r.within_gain:+.2f}$ \\\\")
            first = False
        if task != "synthetic_mean":
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "naturalistic.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/naturalistic.tex")


if __name__ == "__main__":
    main()
