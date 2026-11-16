#!/usr/bin/env python3
"""Analysis of scripts/run_ordered_labels.py: do distinct ordered boundary labels remove the
post-marker blind spot?

Three marker conditions share one grid (running_xor, running_sum_local, random_permutation_digits;
N in {32, 64}; k = 4; 20 trials): "bar" repeats "|", "letter" repeats "a", and "ordered" cycles
a, b, c, ... so every boundary carries a distinct, ordered label. If the blind spot comes from
boundary aliasing -- one symbol standing for every boundary, so the position after one cannot be
told from the position after another -- only "ordered" should repair it. If it comes from any
token interrupting the lookback from input j to input j-1, all three should leave it in place.

Position classes follow Equation 5 as everywhere else in the paper: the crossing position is
i >= k with i mod k = 0, the within-segment positions are i > 0 with i mod k != 0, and the
no-marker run is read at the same indices, so a change is never a comparison between positions.
Standard errors are over trials and unpaired between the marker and no-marker runs.

Outputs: results_clean/_summary/ordered_labels.csv
         paper_tmlr/tables/ordered_labels.tex
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODELS = [("Qwen2.5 32B", "Qwen2.5-32B-Instruct"), ("Gemma 4 31B", "gemma-4-31B-it")]
TASKN = {"running_xor": "Running XOR", "running_sum_local": "Running sum (mod 4)",
         "random_permutation_digits": "Random permutation"}
MARKN = {"bar": "Repeating \\texttt{|}", "letter": "Repeating \\texttt{a}",
         "ordered": "Ordered \\texttt{a}, \\texttt{b}, \\texttt{c}, \\ldots"}
K = 4


def vec(truth, pred):
    p = str(pred)[: len(str(truth))]
    return np.array([i < len(p) and p[i] == c for i, c in enumerate(str(truth))], dtype=float)


def classes(n, k):
    i = np.arange(n)
    return (i >= k) & (i % k == 0), (i > 0) & (i % k != 0)


def cell(df):
    """Mean accuracy overall, at crossing positions and at within-segment positions, per trial."""
    tot, post, within = [], [], []
    for _, r in df.iterrows():
        v = vec(r.Ground_Truth, r.LLM_Output)
        if not len(v):
            continue
        ip, iw = classes(len(v), K)
        tot.append(v.mean())
        if ip.any():
            post.append(v[ip].mean())
        if iw.any():
            within.append(v[iw].mean())

    def f(x):
        return (np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (np.nan, np.nan)

    return f(tot), f(post), f(within)


def main():
    rows = []
    for label, tag in MODELS:
        f = REPO / "results" / "ordered_labels" / f"{tag}.csv"
        if not f.exists():
            print(f"no results for {label}")
            continue
        d = pd.read_csv(f, dtype={"Ground_Truth": str, "LLM_Output": str}, keep_default_na=False)
        d["Seq_Length"] = d.Seq_Length.astype(int)
        for task in TASKN:
            for N in sorted(d.Seq_Length.unique()):
                base = d[(d.Task == task) & (d.Seq_Length == N) & (d.Marker == "base")]
                if base.empty:
                    continue
                (bt, bts), (bp, bps), (bw, bws) = cell(base)
                for marker in MARKN:
                    m = d[(d.Task == task) & (d.Seq_Length == N) & (d.Marker == marker)
                          & (d.Interval.astype(int) == K)]
                    if m.empty:
                        continue
                    (tt, tts), (tp, tps), (tw, tws) = cell(m)

                    def se(a, b):
                        return float(np.sqrt(a ** 2 + b ** 2))

                    rows.append(dict(
                        model=label, task=task, N=N, marker=marker, n=len(m),
                        base=bt, acc=tt, delta=tt - bt, delta_se=se(tts, bts),
                        base_post=bp, post=tp, post_gain=tp - bp, post_se=se(tps, bps),
                        base_within=bw, within=tw, within_gain=tw - bw, within_se=se(tws, bws)))
    if not rows:
        print("nothing to analyse yet")
        return
    o = pd.DataFrame(rows)
    o["post_z"] = o.post_gain / o.post_se.replace(0, np.nan)
    o["within_z"] = o.within_gain / o.within_se.replace(0, np.nan)
    o = o.round(4)
    out = REPO / "results_clean" / "_summary" / "ordered_labels.csv"
    o.to_csv(out, index=False)
    print(f"wrote {out} ({len(o)} rows)")
    pd.set_option("display.width", 250)
    print(o[["model", "task", "N", "marker", "base_post", "post", "post_gain", "post_z",
             "within_gain", "delta"]].to_string(index=False))

    lines = ["\\begin{tabular}{ll l cc cc c}", "\\toprule",
             "Model & Task & Marker & No markers & Markers & Post-marker $\\Delta$ & "
             "Within $\\Delta$ & Overall $\\Delta$ \\\\",
             "\\midrule"]
    for label, _ in MODELS:
        mfirst = True
        for task in TASKN:
            tfirst = True
            for N in (64,):                      # the length the blind spot is measured at
                for marker in MARKN:
                    x = o[(o.model == label) & (o.task == task) & (o.N == N) & (o.marker == marker)]
                    if x.empty:
                        continue
                    r = x.iloc[0]
                    lines.append(
                        f"{label if mfirst else ''} & {TASKN[task] if tfirst else ''} & "
                        f"{MARKN[marker]} & {r.base_post:.2f} & {r.post:.2f} & "
                        f"${r.post_gain:+.2f}$ & ${r.within_gain:+.2f}$ & ${r.delta:+.2f}$ \\\\")
                    mfirst = tfirst = False
            lines.append("\\addlinespace")
    while lines[-1] == "\\addlinespace":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}"]
    t = REPO / "paper_tmlr" / "tables" / "ordered_labels.tex"
    t.write_text("\n".join(lines) + "\n")
    print(f"wrote {t}")


if __name__ == "__main__":
    main()
