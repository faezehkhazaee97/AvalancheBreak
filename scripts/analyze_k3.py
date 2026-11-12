#!/usr/bin/env python3
"""Does the blind spot depend on the spacing continuously? (Reviewer question.)

The main grids include k = 3, excluded from the pooled analyses to keep the spacing grid at
powers of two. Here the per-position comparison of scripts/analyze_per_position.py is repeated
at k = 2, 3 and 4 separately (N <= 64; post-marker and within-segment positions, each against
the no-marker run at the same indices), on the four dependency tasks at the four stronger
models. If the loss is a property of the position right after a marker, k = 3 should look like
its neighbours rather than like an artifact of the power-of-two grid.

Output: results_clean/_summary/blind_spot_by_k.csv, paper_tmlr/tables/blind_spot_by_k.tex
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, LENGTHS  # noqa: E402
from analyze_aligned_scoring import index_vector, clean  # noqa: E402

MODELS = [("Qwen2.5 14B", "Qwen2.5-14B-Instruct"), ("Qwen2.5 32B", "Qwen2.5-32B-Instruct"),
          ("Qwen3.8 27B", "Qwen3.8-27B"), ("Gemma 4 31B", "gemma-4-31B-it")]
TASKS = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]


def per_k(d, k):
    d = d[d.Seq_Length.astype(int).isin(LENGTHS)]
    base = d[d.Method == "Base"]; bnd = d[(d.Method == "Cellular_Bounds") & (d.Interval.astype(int) == k)]
    if bnd.empty:
        return None
    ref = {}
    for _, r in base.iterrows():
        v = index_vector(clean(r.Ground_Truth), clean(r.LLM_Output)); ref.setdefault(len(v), []).append(v)
    ref = {n: np.mean(vs, axis=0) for n, vs in ref.items()}
    post, bpost, within, bwithin = [], [], [], []
    for _, r in bnd.iterrows():
        v = index_vector(clean(r.Ground_Truth), clean(r.LLM_Output)); idx = np.arange(len(v))
        p = (idx > 0) & (idx % k == 0)
        if not p.any() or len(v) not in ref:
            continue
        post.append(v[p].mean()); bpost.append(ref[len(v)][p].mean())
        within.append(v[~p].mean()); bwithin.append(ref[len(v)][~p].mean())
    return dict(n=len(post), post=np.mean(post), post_gain=np.mean(post) - np.mean(bpost),
                within_gain=np.mean(within) - np.mean(bwithin))


def main():
    rows = []
    for task in TASKS:
        for lab, tag in MODELS:
            d = load(task, tag)
            if d is None:
                continue
            for k in (2, 3, 4):
                r = per_k(d, k)
                if r:
                    rows.append(dict(task=task, model=lab, k=k, **r))
    o = pd.DataFrame(rows).round(3)
    o.to_csv(REPO / "results_clean" / "_summary" / "blind_spot_by_k.csv", index=False)
    piv = o.pivot_table(index=["task", "model"], columns="k", values="post_gain")
    print("post-marker change vs no-marker run, by spacing:\n", piv.round(2).to_string())
    print("\nmean over the 16 cells:", piv.mean().round(3).to_dict())
    print("cells where k=3 loss lies between or beyond its neighbours (same sign):",
          int(((np.sign(piv[3]) == np.sign(piv[2])) & (np.sign(piv[3]) == np.sign(piv[4]))).sum()), "of", len(piv))
    lines = ["\\begin{tabular}{ll ccc ccc}", "\\toprule",
             "& & \\multicolumn{3}{c}{Post-marker change} & \\multicolumn{3}{c}{Within-segment change} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}", "Task & Model & $k=2$ & $k=3$ & $k=4$ & $k=2$ & $k=3$ & $k=4$ \\\\", "\\midrule"]
    for task in TASKS:
        for i, (lab, _) in enumerate(MODELS):
            x = o[(o.task == task) & (o.model == lab)].set_index("k")
            if x.empty:
                continue
            c = [f"${x.loc[k, 'post_gain']:+.2f}$" if k in x.index else "---" for k in (2, 3, 4)]
            c += [f"${x.loc[k, 'within_gain']:+.2f}$" if k in x.index else "---" for k in (2, 3, 4)]
            lines.append(("\\task{" + task.replace("_", "\\_") + "}" if i == 0 else "") + f" & {lab} & " + " & ".join(c) + " \\\\")
        if task != TASKS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "blind_spot_by_k.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/blind_spot_by_k.tex")


if __name__ == "__main__":
    main()
