#!/usr/bin/env python3
"""Analysis of scripts/run_marker_sides_forced.py: the input/output marker 2x2 with the output
format enforced by a logits processor rather than requested in the prompt.

Reports, per model, task and cell: the share of generations whose output format came out as
intended, accuracy, the change against the no-marker cell, and the interaction
A11 - A10 - A01 + A00 with a bootstrap interval (trials of the three marked cells are paired by
(N, k, trial); the no-marker cell is resampled independently). Qwen2.5-32B is the paper's main
run; the 7B-class models were added for the review (same script, same cells).

Post-marker accuracy is reported for every cell, not only the ones whose input carries markers.
The scored answer index set is the same in all four cells, P_k = {i : i > 0, i mod k = 0}: with
input markers those indices are the characters whose inputs lie across a marker, and with the
output format forced they are the characters the decoder writes right after a forced delimiter
(the processor emits a bar after every k answer characters, so the two coincide by construction).
The no-marker cell is scored at the same indices, pooled with equal weight over the spacings k
the marked cells use, which is the baseline the other three rows are read against. Within-segment
accuracy (the complement, i > 0 and i mod k != 0) is written to the summary CSV so the
post-marker column can be read against the rest of the same generation.

Outputs: results_clean/_summary/marker_sides_forced.csv,
         paper_tmlr/tables/marker_sides_forced.tex       (main text, Qwen2.5-32B)
         paper_tmlr/tables/marker_sides_forced_full.tex  (same, with the format-realized columns)
         paper_tmlr/tables/marker_sides_forced_7b.tex    (appendix, 7B-class models)
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
TASKS = [("running_xor", "running\\_xor"), ("random_permutation_digits", "random\\_permutation (digits)"),
         ("copy_previous", "copy\\_previous"), ("running_sum_local", "running\\_sum\\_local ($M=4$)")]
CONDS = [("base", "no", "no"), ("in_only", "yes", "no"), ("out_only", "no", "yes"), ("in_out", "yes", "yes")]
B = 10000
RNG = np.random.default_rng(20261111)


def prefix_positions(text, n):
    """Marker offsets within the scored prefix: the first n answer characters and the bars among them
    (bars written after the n-th character do not affect the clipped answer)."""
    line = str(text).strip().split("\n")[0].replace(" ", "")
    out, seen, kept = [], 0, 0
    for c in line:
        if c == "|":
            if seen < n:
                out.append(kept)
            kept += 1
        else:
            if seen >= n:
                break
            seen += 1; kept += 1
    while out and out[-1] == kept - 1 and seen >= n:   # a bar right after the n-th character
        out.pop()
    return out


def vec(t, p):
    p = "" if pd.isna(p) else str(p)
    return np.array([i < len(p) and p[i] == c for i, c in enumerate(t)], dtype=float)


def split_acc(g, ks):
    """Accuracy at the post-marker indices P_k = {i > 0 : i mod k = 0} and at the rest, one pair
    of values per scored generation. Generations carry their own k; the no-marker cell has k = 0
    recorded and is scored at every spacing in `ks`, each spacing weighted equally, so it covers
    the same indices as the marked cells it is the baseline for (the spacings are taken in order,
    all generations at one spacing before the next, which fixes the bootstrap's resampling unit)."""
    post, within = [], []
    spacings = ks if (g.k == 0).all() else [None]     # [None] means "each generation's own k"
    for k0 in spacings:
        for r in g.itertuples():
            k = k0 if k0 is not None else r.k
            v = vec(r.Ground_Truth, r.LLM_Output)
            idx = np.arange(len(v))
            m = (idx > 0) & (idx % k == 0)
            if m.any():
                post.append(v[m].mean())
                within.append(v[(idx > 0) & ~m].mean())
    return np.array(post), np.array(within)


def did_ci(D, bD, rng):
    """(post - within) of a cell minus (post - within) of the no-marker cell, with a percentile
    bootstrap over generations: the cell and the baseline are resampled independently, as the
    interaction bootstrap resamples the no-marker cell independently."""
    reps = np.array([D[rng.integers(0, len(D), len(D))].mean() - bD[rng.integers(0, len(bD), len(bD))].mean()
                     for _ in range(B)])
    return D.mean() - bD.mean(), *np.percentile(reps, [2.5, 97.5])


def analyze(label, d):
    d = d.assign(k=d.Interval.astype(int), N=d.Seq_Length.astype(int), t=d.Trial.astype(int),
                 TAR=d.TAR.astype(float))
    rows, inter = [], {}
    did_rng = np.random.default_rng(7)               # own stream, so the DiD intervals do not
    for task, _ in TASKS:                            # move when the interaction bootstrap changes
        x = d[d.Task == task]
        base = x[x.Condition == "base"]
        if base.empty:
            continue
        ks = sorted({int(v) for v in x.k if v > 0})      # the spacings the marked cells use
        bP, bW = split_acc(base, ks)
        bD = bP - bW
        for cond, hi, ho in CONDS:
            g = x[x.Condition == cond]
            want = cond in ("out_only", "in_out")
            ok = np.mean([(marker_positions(r.Raw_Model_Output) == expected_positions(len(r.Ground_Truth), r.k))
                          if want else (len(marker_positions(r.Raw_Model_Output)) == 0) for r in g.itertuples()])
            okp = np.array([(prefix_positions(r.Raw_Model_Output, len(r.Ground_Truth)) == expected_positions(len(r.Ground_Truth), r.k))
                            if want else (len(prefix_positions(r.Raw_Model_Output, len(r.Ground_Truth))) == 0) for r in g.itertuples()])
            acc_ok = g.TAR.to_numpy()[okp].mean() if okp.any() else np.nan
            P, W = split_acc(g, ks)
            did, lo_d, hi_d = ((np.nan,) * 3 if cond == "base" else did_ci(P - W, bD, did_rng))
            rows.append(dict(model=label, task=task, cond=cond, markers_in=hi, markers_out=ho, n=len(g),
                             acc=g.TAR.mean(), se=g.TAR.sem(), delta=g.TAR.mean() - base.TAR.mean(),
                             format_ok=ok, format_prefix=okp.mean(), acc_compliant=acc_ok,
                             post=P.mean(), within=W.mean(), post_within=P.mean() - W.mean(),
                             did=did, did_lo=lo_d, did_hi=hi_d))
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
    names = {"base": "Neither side", "in_only": "Input only", "out_only": "Output only", "in_out": "Both sides"}

    def table(models, path, full=True):
        """full=False drops the two columns that carry no information in these runs: the scored
        first N characters comply in essentially every generation, which makes `Acc. if so` a copy
        of `Accuracy`. The full version keeps them for the appendix. Every version carries the
        post-marker and within-segment columns and their difference-in-differences against the
        no-marker cell, which is the contrast Section~\\ref{sec:sides} rests on."""
        nc = 12 if full else 10
        lines = ["\\begin{tabular}{ll cc cc ccc cc l}" if full else "\\begin{tabular}{ll cc cc c cc l}",
                 "\\toprule",
                 ("& & \\multicolumn{2}{c}{Markers} & & & \\multicolumn{3}{c}{Format realized}"
                  " & \\multicolumn{2}{c}{Accuracy by position} & \\\\" if full else
                  "& & \\multicolumn{2}{c}{Markers} & & & & \\multicolumn{2}{c}{Accuracy by position} & \\\\"),
                 ("\\cmidrule(lr){3-4}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}" if full
                  else "\\cmidrule(lr){3-4}\\cmidrule(lr){8-9}"),
                 "Task & Condition & Input & Output & Accuracy & Change & Full output & "
                 + ("First $N$ & Acc.\\ if so & " if full else "")
                 + "Post-marker & Within segment & (Post $-$ within) $-$ base \\\\", "\\midrule"]
        blocks = [(m, t) for m in models for t, _ in TASKS if (m, t) in inter]
        for bi, (label, task) in enumerate(blocks):
            lab = dict(TASKS)[task]
            if len(models) > 1 and (bi == 0 or blocks[bi - 1][0] != label):
                lines.append(f"\\multicolumn{{{nc}}}{{l}}{{\\emph{{{label}}}}} \\\\")
            for i, (cond, hi_, ho_) in enumerate(CONDS):
                r = o[(o.model == label) & (o.task == task) & (o.cond == cond)].iloc[0]
                ch = "---" if cond == "base" else f"${r.delta:+.2f}$"
                post = "---" if np.isnan(r.post) else f"{r.post:.2f}"
                wit = "---" if np.isnan(r.within) else f"{r.within:.2f}"
                did = ("---" if np.isnan(r.did)
                       else f"${r.did:+.2f}$ $[{r.did_lo:+.2f}, {r.did_hi:+.2f}]$")
                cells = ["\\task{" + lab + "}" if i == 0 else "", names[cond], hi_, ho_,
                         f"{r.acc:.2f} $\\pm$ {1.96 * r.se:.2f}", ch,
                         f"{r.format_ok:.0%}".replace("%", "\\%")]
                if full:
                    cells += [f"{r.format_prefix:.0%}".replace("%", "\\%"),
                              "---" if np.isnan(r.acc_compliant) else f"{r.acc_compliant:.2f}"]
                lines.append(" & ".join(cells + [post, wit, did]) + " \\\\")
            p, lo, hi = inter[(label, task)]
            lines.append(f"\\multicolumn{{{nc}}}{{l}}{{\\quad Interaction ${p:+.2f}$, 95\\% interval $[{lo:+.2f}, {hi:+.2f}]$}} \\\\")
            if bi != len(blocks) - 1:
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (REPO / "paper_tmlr" / "tables" / path).write_text("\n".join(lines) + "\n")
        print("Wrote paper_tmlr/tables/" + path)

    table([MODELS[0][0]], "marker_sides_forced.tex", full=False)      # Qwen2.5-32B, main text
    table([MODELS[0][0]], "marker_sides_forced_full.tex")             # the same with every column
    table([m for m, _ in MODELS[1:]], "marker_sides_forced_7b.tex")   # 7B-class models, appendix


if __name__ == "__main__":
    main()
