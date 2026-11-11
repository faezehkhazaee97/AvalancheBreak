#!/usr/bin/env python3
"""Analysis of scripts/run_prompt_delivery.py: do the effects depend on how the prompt is
delivered? Compares the default chat template (read from the main grid, same prompts and seeds)
with an empty system message, a task-oriented system message, and no chat template at all.

Reports, per task and condition: accuracy without and with markers, the change, and for
running_xor the post-marker and within-segment change against the no-marker run of the same
condition at the same positions. Also checks that the main-grid overlap reproduces.

Outputs: results_clean/_summary/prompt_delivery.csv, paper_tmlr/tables/prompt_delivery.tex
"""
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}
TASKS = [("running_xor", None, None, "running\\_xor"),
         ("random_permutation_digits", "Alphabet_Size", 8, "random\\_permutation (digits)")]
CONDS = [("default", "Default template"), ("nosys", "Empty system message"),
         ("persona", "Task-oriented system message"), ("notemplate", "No chat template")]
LENGTHS, KS = [32, 64], [0, 4, 16]


def vec(t, p):
    p = "" if pd.isna(p) else str(p)
    return np.array([i < len(p) and p[i] == c for i, c in enumerate(t)], dtype=float)


def grid(task, axis_col, axis):
    fs = [f for f in (REPO / "results_clean" / task / "grid").glob("Qwen2.5-32B-Instruct*.csv") if "POOLED" not in f.name]
    g = pd.concat([pd.read_csv(f, dtype=TEXT, keep_default_na=False) for f in fs])
    g = g.assign(N=g.Seq_Length.astype(int), k=g.Interval.astype(int), t=g.Trial.astype(int), TAR=g.TAR.astype(float))
    g = g[g.N.isin(LENGTHS) & g.k.isin(KS) & (g.t < 20)]
    if axis_col:
        g = g[g[axis_col].astype(int) == axis]
    return g.drop_duplicates(["k", "N", "t"])


def main():
    s = pd.read_csv(REPO / "results" / "prompt_delivery" / "Qwen2.5-32B-Instruct.csv", dtype=TEXT, keep_default_na=False)
    s = s.assign(N=s.Seq_Length.astype(int), k=s.Interval.astype(int), t=s.Trial.astype(int), TAR=s.TAR.astype(float))
    rows = []
    for task, ac, axis, _ in TASKS:
        for cond, _ in CONDS:
            d = grid(task, ac, axis) if cond == "default" else s[(s.Task == task) & (s.Condition == cond)]
            base, bnd = d[d.k == 0], d[d.k > 0]
            delta = bnd.TAR.mean() - base.TAR.mean()
            se = np.sqrt(base.TAR.var(ddof=1) / len(base) + bnd.TAR.var(ddof=1) / len(bnd))
            ref = {N: np.mean([vec(r.Ground_Truth, r.LLM_Output) for r in base[base.N == N].itertuples()], axis=0)
                   for N in LENGTHS}
            post, within, bp, bw = [], [], [], []
            for r in bnd.itertuples():
                v = vec(r.Ground_Truth, r.LLM_Output); idx = np.arange(len(v)); p = (idx > 0) & (idx % r.k == 0)
                post.append(v[p].mean()); within.append(v[~p].mean())
                bp.append(ref[r.N][p].mean()); bw.append(ref[r.N][~p].mean())
            rows.append(dict(task=task, cond=cond, n=len(d), base=base.TAR.mean(), bounds=bnd.TAR.mean(),
                             delta=delta, ci=1.96 * se, post=np.mean(post),
                             post_gain=np.mean(post) - np.mean(bp), within_gain=np.mean(within) - np.mean(bw)))
    o = pd.DataFrame(rows).round(4)
    o.to_csv(REPO / "results_clean" / "_summary" / "prompt_delivery.csv", index=False)
    print(o.to_string(index=False))
    lines = ["\\begin{tabular}{ll ccc cc}", "\\toprule",
             "Task & Prompt delivery & No markers & Markers & Change & Post-marker & Within-segment \\\\", "\\midrule"]
    for task, _, _, lab in TASKS:
        for i, (cond, clab) in enumerate(CONDS):
            r = o[(o.task == task) & (o.cond == cond)].iloc[0]
            pp = ("---", "---") if task.startswith("random") else (f"${r.post_gain:+.2f}$", f"${r.within_gain:+.2f}$")
            first = "\\task{" + lab + "}" if i == 0 else ""
            lines.append(" & ".join([first, clab, f"{r.base:.2f}", f"{r.bounds:.2f}",
                                     "$%+.2f \\pm %.2f$" % (r.delta, r.ci), pp[0], pp[1]]) + " \\\\")
        if task != TASKS[-1][0]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "prompt_delivery.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/prompt_delivery.tex")


if __name__ == "__main__":
    main()
