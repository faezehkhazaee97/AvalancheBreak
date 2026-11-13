#!/usr/bin/env python3
"""Analysis of scripts/run_marker_sides_forced.py: the input/output marker 2x2 with the output
format enforced by a logits processor rather than requested in the prompt.

Reports, per model, task and cell: the share of generations whose output format came out as
intended, accuracy, the change against the no-marker cell, and the interaction
A11 - A10 - A01 + A00 with a bootstrap interval (trials of the three marked cells are paired by
(N, k, trial); the no-marker cell is resampled independently). Also gives post-marker accuracy
where the input carries markers. Qwen2.5-32B is the paper's main run; the 7B-class models were
added for the review (same script, same cells).

Outputs: results_clean/_summary/marker_sides_forced.csv, paper_tmlr/tables/marker_sides_forced.tex
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_compliance import marker_positions, expected_positions  # noqa: E402

TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}
MODELS = [("Qwen2.5 32B", "Qwen2.5-32B-Instruct"), ("Qwen2.5 7B", "Qwen2.5-7B-Instruct"),
          ("Mistral 7B", "Mistral-7B-Instruct-v0.3")]
TASKS = [("running_xor", "running\\_xor"), ("random_permutation_digits", "random\\_permutation (digits)")]
CONDS = [("base", "no", "no"), ("in_only", "yes", "no"), ("out_only", "no", "yes"), ("in_out", "yes", "yes")]
B = 10000
RNG = np.random.default_rng(20261111)


def vec(t, p):
    p = "" if pd.isna(p) else str(p)
    return np.array([i < len(p) and p[i] == c for i, c in enumerate(t)], dtype=float)


def analyze(label, d):
    d = d.assign(k=d.Interval.astype(int), N=d.Seq_Length.astype(int), t=d.Trial.astype(int),
                 TAR=d.TAR.astype(float))
    rows, inter = [], {}
    for task, _ in TASKS:
        x = d[d.Task == task]
        base = x[x.Condition == "base"]
        if base.empty:
            continue
        for cond, hi, ho in CONDS:
            g = x[x.Condition == cond]
            want = cond in ("out_only", "in_out")
            ok = np.mean([(marker_positions(r.Raw_Model_Output) == expected_positions(len(r.Ground_Truth), r.k))
                          if want else (len(marker_positions(r.Raw_Model_Output)) == 0) for r in g.itertuples()])
            post = np.nan
            if hi == "yes":
                pv = []
                for r in g.itertuples():
                    v = vec(r.Ground_Truth, r.LLM_Output); idx = np.arange(len(v))
                    m = (idx > 0) & (idx % r.k == 0)
                    if m.any():
                        pv.append(v[m].mean())
                post = np.mean(pv)
            rows.append(dict(model=label, task=task, cond=cond, markers_in=hi, markers_out=ho, n=len(g),
                             acc=g.TAR.mean(), se=g.TAR.sem(), delta=g.TAR.mean() - base.TAR.mean(),
                             format_ok=ok, post=post))
        piv = {c: x[x.Condition == c].set_index(["N", "k", "t"]).TAR for c in ("in_out", "in_only", "out_only")}
        keys = piv["in_out"].index.intersection(piv["in_only"].index).intersection(piv["out_only"].index)
        M = np.column_stack([piv[c].loc[keys].to_numpy() for c in ("in_out", "in_only", "out_only")])
        a00 = base.TAR.to_numpy()
        point = M[:, 0].mean() - M[:, 1].mean() - M[:, 2].mean() + a00.mean()
        reps = np.array([(lambda r, b: r[:, 0].mean() - r[:, 1].mean() - r[:, 2].mean() + b.mean())(
            M[RNG.integers(0, len(M), len(M))], a00[RNG.integers(0, len(a00), len(a00))]) for _ in range(B)])
        inter[(label, task)] = (point, *np.percentile(reps, [2.5, 97.5]))
    return rows, inter


def main():
    rows, inter = [], {}
    for label, tag in MODELS:
        f = REPO / "results" / "marker_sides_forced" / f"{tag}.csv"
        if not f.exists():
            continue
        r, i = analyze(label, pd.read_csv(f, dtype=TEXT, keep_default_na=False))
        rows += r
        inter.update(i)
    o = pd.DataFrame(rows).round(4)
    o.to_csv(REPO / "results_clean" / "_summary" / "marker_sides_forced.csv", index=False)
    print(o.to_string(index=False))
    for (m, t), (p, lo, hi) in inter.items():
        print(f"interaction {m} {t}: {p:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]")
    lines = ["\\begin{tabular}{ll cc cc cc}", "\\toprule",
             "& & \\multicolumn{2}{c}{Markers} & & & & \\\\", "\\cmidrule(lr){3-4}",
             "Task & Condition & Input & Output & Accuracy & Change & Format & Post-marker \\\\", "\\midrule"]
    names = {"base": "Neither side", "in_only": "Input only", "out_only": "Output only", "in_out": "Both sides"}
    blocks = [(m, t) for m, _ in MODELS for t, _ in TASKS if (m, t) in inter]
    for bi, (label, task) in enumerate(blocks):
        lab = dict(TASKS)[task]
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{label}}}}} \\\\" if bi == 0 or blocks[bi - 1][0] != label else "")
        for i, (cond, hi_, ho_) in enumerate(CONDS):
            r = o[(o.model == label) & (o.task == task) & (o.cond == cond)].iloc[0]
            ch = "---" if cond == "base" else f"${r.delta:+.2f}$"
            post = "---" if np.isnan(r.post) else f"{r.post:.2f}"
            lines.append(" & ".join(["\\task{" + lab + "}" if i == 0 else "", names[cond], hi_, ho_,
                                     f"{r.acc:.2f} $\\pm$ {1.96 * r.se:.2f}", ch,
                                     f"{r.format_ok:.0%}".replace("%", "\\%"), post]) + " \\\\")
        p, lo, hi = inter[(label, task)]
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\quad Interaction ${p:+.2f}$, 95\\% interval $[{lo:+.2f}, {hi:+.2f}]$}} \\\\")
        if bi != len(blocks) - 1:
            lines.append("\\midrule")
    lines = [l for l in lines if l != ""]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "marker_sides_forced.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/marker_sides_forced.tex")


if __name__ == "__main__":
    main()
