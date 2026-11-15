#!/usr/bin/env python3
"""Do markers change how errors cluster along the sequence? (Reviewer question.)

For every stored generation at N = 64 of the four stronger open models, the per-position error
indicator e_i (1 = wrong) is computed under the paper's index scoring and under the edit-distance
alignment of scripts/analyze_aligned_scoring.py. Three quantities are pooled over generations:

  rate   the error rate, mean of e_i
  p11    P(e_{i+1} = 1 | e_i = 1)   errors following an error (persistence)
  p01    P(e_{i+1} = 1 | e_i = 0)   errors following a correct position
  run    mean length of a maximal run of consecutive errors (total errors / number of runs)

grouped by marker density: no markers (k = 0), a marker after every character (k = 1), and sparse
markers (k in {2, 4, 8, 16}). Under index scoring a single dropped or inserted character makes
every later position wrong, so p11 near 1 there measures cascading misalignment; under aligned
scoring it measures genuine clustering of local errors. Formats: spaced letters and digits
(position-local tasks pooled, every alphabet) and the four dependency tasks pooled.

Conventions. Markers and spaces are stripped before scoring, so e_i runs over the output characters
only and every adjacent pair (i, i+1) of a generation enters p11 and p01, including pairs that
straddle a marker; pairs never span two generations. A run cut off by the end of the sequence is
counted with the length it has inside the sequence (no correction for censoring). Under index
scoring a prediction shorter than the truth makes every missing position an error, so a truncated
or desynchronized output ends in one run that reaches the end.

Post-marker errors (dependency tasks, sparse markers). For the post-marker positions
(i > 0, i mod k = 0) the script also counts how many errors are isolated (e_{i-1} = 0 and
e_{i+1} = 0) and recomputes p11 and p01 without the pairs that touch a post-marker position,
to separate the blind spot's single-position errors from the clustering of the other errors.

Outputs: results_clean/_summary/error_autocorrelation.csv,
         results_clean/_summary/error_autocorrelation_postmarker.csv, paper_tmlr/tables/autocorr.tex
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


def postmarker(errs, ks):
    """Isolation of post-marker errors and p11 / p01 without the pairs touching a post-marker position."""
    n_post = n_err = n_iso = 0
    a_all, b_all, a_ex, b_ex = [], [], [], []
    for v, k in zip(errs, ks):
        idx = np.arange(len(v))
        post = (idx > 0) & (idx % k == 0) & (idx < len(v) - 1)
        n_post += int(post.sum())
        e = post & (v == 1)
        n_err += int(e.sum())
        n_iso += int((e & np.r_[False, v[:-1] == 0] & np.r_[v[1:] == 0, False]).sum())
        keep = ~(post[:-1] | post[1:] | ((idx[:-1] > 0) & (idx[:-1] % k == 0)) | ((idx[1:] > 0) & (idx[1:] % k == 0)))
        a_all.append(v[:-1]); b_all.append(v[1:]); a_ex.append(v[:-1][keep]); b_ex.append(v[1:][keep])
    a, b, ax, bx = map(np.concatenate, (a_all, b_all, a_ex, b_ex))
    return dict(post_positions=n_post, post_errors=n_err, post_error_rate=n_err / max(n_post, 1),
                isolated=n_iso, isolated_share=n_iso / max(n_err, 1),
                p11_all=b[a == 1].mean(), p01_all=b[a == 0].mean(),
                p11_excl_post=bx[ax == 1].mean(), p01_excl_post=bx[ax == 0].mean())


def main():
    rows, prows = [], []
    for label, tag in MODELS:
        for fmt, tasks in FORMATS:
            vecs = {(g, s): [] for g, _ in GROUPS for s in ("index", "aligned")}
            kvec = {g: [] for g, _ in GROUPS}
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
                    kvec[g].append(k)
                    vecs[(g, "index")].append(1 - index_vector(t, p))
                    vecs[(g, "aligned")].append(1 - aligned_vector(t, p))
            for (g, s), v in vecs.items():
                if v:
                    rows.append(dict(model=label, format=fmt, group=g, scorer=s, **stats(v)))
                if v and g == "sparse":
                    prows.append(dict(model=label, format=fmt, scorer=s, **postmarker(v, kvec[g])))
            print("done", label, fmt, flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(REPO / "results_clean" / "_summary" / "error_autocorrelation.csv", index=False)
    print(out.pivot_table(index=["model", "format", "scorer"], columns="group", values=["rate", "p11", "p01", "run"])
          .round(3).to_string())
    for scorer in ("index", "aligned"):
        x = out[out.scorer == scorer]
        for fmt, _t in FORMATS:
            y = x[x.format == fmt].set_index("group").run
            print(f"mean error-run length, {scorer} scoring, {fmt}: no markers {y.loc['none'].min():.2f} to "
                  f"{y.loc['none'].max():.2f}; k = 1 {y.loc['k1'].min():.2f} to {y.loc['k1'].max():.2f}; "
                  f"sparse {y.loc['sparse'].min():.2f} to {y.loc['sparse'].max():.2f}")
    pm = pd.DataFrame(prows).round(4)
    pm.to_csv(REPO / "results_clean" / "_summary" / "error_autocorrelation_postmarker.csv", index=False)
    print("\nPost-marker errors at sparse markers (N = 64): isolated = correct at i-1 and i+1")
    print(pm.to_string(index=False))

    lines = [r"\begin{tabular}{ll cccc cccc cccc}", r"\toprule",
             r"& & \multicolumn{4}{c}{No markers} & \multicolumn{4}{c}{$k = 1$} & \multicolumn{4}{c}{$k \in \{2,4,8,16\}$} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
             r"Model & Format" + r" & rate & $p_{1|1}$ & $p_{1|0}$ & run" * 3 + r" \\",
             r"\midrule"]
    for scorer in ("index", "aligned"):
        lines.append(r"\multicolumn{14}{l}{\emph{" + ("Index scoring" if scorer == "index" else "Aligned scoring") + r"}} \\")
        for label, _ in MODELS:
            first = True
            for fmt, _t in FORMATS:
                cells = []
                for g, _k in GROUPS:
                    x = out[(out.model == label) & (out.format == fmt) & (out.group == g) & (out.scorer == scorer)]
                    cells += ["---"] * 4 if x.empty else [f"{x.rate.iloc[0]:.2f}", f"{x.p11.iloc[0]:.2f}", f"{x.p01.iloc[0]:.2f}",
                                                          f"{x.run.iloc[0]:.1f}"]
                lines.append(f"{label if first else ''} & {fmt} & " + " & ".join(cells) + r" \\")
                first = False
        if scorer == "index":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "autocorr.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/autocorr.tex")


if __name__ == "__main__":
    main()
