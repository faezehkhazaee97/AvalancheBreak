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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS as ALL_MODELS, LENGTHS, horizon, position_classes  # noqa: E402
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


def reset_truth(task, shown):
    """Ground truth if the running state restarted at every marker (segments treated as fresh starts)."""
    import importlib
    e = importlib.import_module(f"tasks.{task}.engine")
    fn = e._depth if task == "bracket_depth" else e._checksum
    return "".join(fn(seg) for seg in shown.split("|"))


def reset_consistency(task, d):
    """Over marked rows (k >= 2, N <= 64): index accuracy against the true answer and against the
    reset answer, at post-marker and within-segment positions."""
    import numpy as np
    from analyze_aligned_scoring import index_vector, clean
    d = d[(d.Method == "Cellular_Bounds") & d.Interval.astype(int).isin([2, 4, 8, 16]) & d.Seq_Length.astype(int).isin(LENGTHS)]
    out = {"post_true": [], "post_reset": [], "within_true": [], "within_reset": [],
           "post_coincide": [], "post_true_diff": [], "post_reset_diff": []}
    for _, r in d.iterrows():
        k = int(r.Interval)
        pred = clean(r.LLM_Output)
        truth, rtruth = clean(r.Ground_Truth), reset_truth(task, r.Input_Given)
        vt = index_vector(truth, pred)
        vr = index_vector(rtruth, pred)
        h = horizon(task)
        idx = np.arange(len(vt)); p, w = position_classes(idx, k, h)
        if not p.any():
            continue
        out["post_true"].append(vt[p].mean()); out["post_reset"].append(vr[p].mean())
        # The two answers agree wherever the prefix state before the marker is already 0 mod M, and
        # a position where they agree counts for both columns. Record how often that happens and
        # the match rates on the positions where the two answers actually differ, which is the only
        # place the comparison can distinguish a reset from the true rule.
        n = min(len(truth), len(rtruth))
        same = np.array([truth[i] == rtruth[i] for i in range(n)])
        pp = p[:n]
        if pp.any():
            out["post_coincide"].append(same[pp].mean())
            diff = pp & ~same
            if diff.any():
                out["post_true_diff"].append(vt[:n][diff].mean())
                out["post_reset_diff"].append(vr[:n][diff].mean())
        if k > h and w.any():
            out["within_true"].append(vt[w].mean()); out["within_reset"].append(vr[w].mean())
    return {k: (float(np.mean(v)) if len(v) else float("nan")) for k, v in out.items()}


def row(task, label, d):
    s = summarize(d, index_vector, horizon(task))
    r = dict(task=task, model=label, n=len(d), base=s["base"], delta=pooled_delta(d),
             post=s["post"], post_gain=s["post_gain"], within_gain=s["within_gain"])
    if task in ("bracket_depth", "running_checksum"):
        r.update(reset_consistency(task, d))
    return r


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
    lines = ["\\begin{tabular}{ll cccc cc cc}", "\\toprule",
             "& & & & & & \\multicolumn{2}{c}{Matches true answer} & \\multicolumn{2}{c}{Matches reset answer} \\\\", "\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
             "Task & Model & No-marker acc. & $\\Delta$ & Post-marker & Within-segment & Post & Within & Post & Within \\\\", "\\midrule"]
    for task in [t for t, _ in NAT] + ["synthetic_mean"]:
        first = True
        for label, _ in ALL_MODELS:
            if label == "Qwen2.5 7B":
                continue    # at chance on both natural tasks (0.30 and 0.22 against 0.25 and 0.10); disclosed in the text
            x = out[(out.task == task) & (out.model == label)]
            if x.empty:
                continue
            r = x.iloc[0]
            tlab = names[task] if first else ""
            rc = (f"{r.post_true:.2f} & {r.within_true:.2f} & {r.post_reset:.2f} & {r.within_reset:.2f}" if task != "synthetic_mean" else "-- & -- & -- & --")
            lines.append(f"{tlab} & {label} & {r.base:.2f} & ${r.delta:+.2f}$ & ${r.post_gain:+.2f}$ & ${r.within_gain:+.2f}$ & {rc} \\\\")
            first = False
        if task != "synthetic_mean":
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "naturalistic.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/naturalistic.tex")


if __name__ == "__main__":
    main()
