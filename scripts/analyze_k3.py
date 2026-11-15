#!/usr/bin/env python3
"""Does the blind spot depend on the spacing continuously? (Reviewer question.)

The main grids include k = 3, excluded from the pooled analyses to keep the spacing grid at
powers of two. Here the per-position comparison of scripts/analyze_per_position.py is repeated
at k = 2, 3, 4, 5 and 6 separately (N <= 64; k = 5 and 6 come from the
<task>_finek runs, which reuse the main grid's seeds and no-marker reference; post-marker and within-segment positions, each against
the no-marker run at the same indices), on the four dependency tasks at the four stronger
models. If the loss is a property of the position right after a marker, k = 3 should look like
its neighbours rather than like an artifact of the power-of-two grid.

sliding_window_max reads the two previous inputs (window of 3), so the position one after a
marker also has an input across it. For that task the positions are also split into
  first     i mod k = 0, i > 0   (the post-marker position above)
  second    i mod k = 1, i > k   (one after the post-marker position; its oldest input is across the marker)
  internal  i mod k >= 2         (no input across a marker; exists only for k >= 3)
Positions 0 and 1 (start of the first segment, no marker before them) are in none of the three.
Each class is compared with the no-marker run at the same indices, per k = 2 to 6 and pooled
over k in {2, 4, 8, 16} (as in scripts/analyze_per_position.py).

Output: results_clean/_summary/blind_spot_by_k.csv, results_clean/_summary/blind_spot_window.csv,
        paper_tmlr/tables/blind_spot_by_k.tex, paper_tmlr/tables/blind_spot_window.tex
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, LENGTHS, KS  # noqa: E402
from analyze_aligned_scoring import index_vector, clean  # noqa: E402

MODELS = [("Qwen2.5 7B", "Qwen2.5-7B-Instruct"), ("Qwen2.5 14B", "Qwen2.5-14B-Instruct"), ("Qwen2.5 32B", "Qwen2.5-32B-Instruct"),
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


def signed(x):
    x = round(float(x), 2) + 0.0  # avoid "-0.00" and "+0.00"
    return "$0.00$" if x == 0 else f"${'+' if x > 0 else '-'}{abs(x):.2f}$"


WINDOW_CLASSES = [("first", "First after marker"), ("second", "Second after marker"), ("internal", "Strictly internal")]


def window_classes(idx, k):
    return {"first": (idx > 0) & (idx % k == 0),
            "second": (idx > k) & (idx % k == 1),
            "internal": idx % k >= 2}


def window_split(d, ks):
    """sliding_window_max: accuracy and change vs the no-marker run at the same indices for the three
    position classes, pooling the marker rows with k in ks (rows are the unit, as in per_k)."""
    d = d[d.Seq_Length.astype(int).isin(LENGTHS)]
    base = d[d.Method == "Base"]; bnd = d[(d.Method == "Cellular_Bounds") & d.Interval.astype(int).isin(ks)]
    if bnd.empty:
        return None
    ref = {}
    for _, r in base.iterrows():
        v = index_vector(clean(r.Ground_Truth), clean(r.LLM_Output)); ref.setdefault(len(v), []).append(v)
    ref = {n: np.mean(vs, axis=0) for n, vs in ref.items()}
    acc = {c: [] for c, _ in WINDOW_CLASSES}; bacc = {c: [] for c, _ in WINDOW_CLASSES}
    for _, r in bnd.iterrows():
        k = int(r.Interval)
        v = index_vector(clean(r.Ground_Truth), clean(r.LLM_Output)); idx = np.arange(len(v))
        if not ((idx > 0) & (idx % k == 0)).any() or len(v) not in ref:
            continue
        for c, m in window_classes(idx, k).items():
            if m.any():
                acc[c].append(v[m].mean()); bacc[c].append(ref[len(v)][m].mean())
    out = {}
    for c, _ in WINDOW_CLASSES:
        out[c + "_n"] = len(acc[c])
        out[c] = np.mean(acc[c]) if acc[c] else np.nan
        out[c + "_change"] = np.mean(acc[c]) - np.mean(bacc[c]) if acc[c] else np.nan
    return out


def window_table():
    task = "sliding_window_max"
    rows = []
    for lab, tag in MODELS:
        d = load(task, tag)
        if d is None:
            continue
        fine = load(task + "_finek", tag)
        for k in (2, 3, 4, 5, 6):
            if k in (5, 6):
                if fine is None:
                    continue
                r = window_split(pd.concat([d[d.Method == "Base"], fine[fine.Interval.astype(int) == k]], ignore_index=True), [k])
            else:
                r = window_split(d, [k])
            if r:
                rows.append(dict(model=lab, k=str(k), **r))
        r = window_split(d, KS)
        if r:
            rows.append(dict(model=lab, k="pooled", **r))
    o = pd.DataFrame(rows).round(3)  # same rounding as blind_spot_by_k, so shared cells print alike
    o.to_csv(REPO / "results_clean" / "_summary" / "blind_spot_window.csv", index=False)
    print("\nsliding_window_max: change vs the no-marker run at the same indices, by position class")
    print(o.pivot_table(index="model", columns="k", values=[c + "_change" for c, _ in WINDOW_CLASSES], sort=False).round(3).to_string())
    strong = ["Qwen2.5 14B", "Qwen2.5 32B", "Qwen3.8 27B", "Gemma 4 31B"]
    p = o[(o.k == "pooled") & o.model.isin(strong)]
    print("pooled k in {2,4,8,16}, mean over the four stronger models: " +
          ", ".join(f"{c} {p[c + '_change'].mean():+.3f} (acc {p[c].mean():.3f})" for c, _ in WINDOW_CLASSES))
    KCOLS = ["2", "3", "4", "5", "6", "pooled"]
    lines = ["\\begin{tabular}{ll cccccc}", "\\toprule",
             "& & \\multicolumn{5}{c}{Spacing $k$} & Pooled \\\\", "\\cmidrule(lr){3-7}",
             "Model & Position & " + " & ".join(f"${k}$" for k in KCOLS[:-1]) + " & $k \\in \\{2,4,8,16\\}$ \\\\", "\\midrule"]
    for i, (lab, _) in enumerate(MODELS):
        x = o[o.model == lab].set_index("k")
        if x.empty:
            continue
        for j, (c, name) in enumerate(WINDOW_CLASSES):
            cells = [signed(x.loc[k, c + "_change"]) if k in x.index and not np.isnan(x.loc[k, c + "_change"]) else "---"
                     for k in KCOLS]
            lines.append((lab if j == 0 else "") + f" & {name} & " + " & ".join(cells) + " \\\\")
        if i < len(MODELS) - 1:
            lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "blind_spot_window.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/blind_spot_window.tex (no strictly internal positions at k = 2: shown as ---)")


def main():
    rows = []
    for task in TASKS:
        for lab, tag in MODELS:
            d = load(task, tag)
            if d is None:
                continue
            fine = load(task + "_finek", tag)
            for k in (2, 3, 4, 5, 6):
                if k in (5, 6):
                    if fine is None:
                        continue
                    r = per_k(pd.concat([d[d.Method == "Base"], fine[fine.Interval.astype(int) == k]], ignore_index=True), k)
                else:
                    r = per_k(d, k)
                if r:
                    rows.append(dict(task=task, model=lab, k=k, **r))
    o = pd.DataFrame(rows).round(3)
    o.to_csv(REPO / "results_clean" / "_summary" / "blind_spot_by_k.csv", index=False)
    piv = o.pivot_table(index=["task", "model"], columns="k", values="post_gain")
    print("post-marker change vs no-marker run, by spacing:\n", piv.round(2).to_string())
    print(f"\nmean over the {len(piv)} cells:", piv.mean().round(3).to_dict())
    print("cells where k=3 loss lies between or beyond its neighbours (same sign):",
          int(((np.sign(piv[3]) == np.sign(piv[2])) & (np.sign(piv[3]) == np.sign(piv[4]))).sum()), "of", len(piv))
    KS = (2, 3, 4, 5, 6)
    lines = ["\\begin{tabular}{ll ccccc ccccc}", "\\toprule",
             "& & \\multicolumn{5}{c}{Post-marker change} & \\multicolumn{5}{c}{Within-segment change} \\\\",
             "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}", "Task & Model & " + " & ".join(f"$k={k}$" for k in KS) + " & " + " & ".join(f"$k={k}$" for k in KS) + " \\\\", "\\midrule"]
    for task in TASKS:
        for i, (lab, _) in enumerate(MODELS):
            x = o[(o.task == task) & (o.model == lab)].set_index("k")
            if x.empty:
                continue
            c = [signed(x.loc[k, 'post_gain']) if k in x.index else "---" for k in KS]
            c += [signed(x.loc[k, 'within_gain']) if k in x.index else "---" for k in KS]
            lines.append(("\\task{" + task.replace("_", "\\_") + "}" if i == 0 else "") + f" & {lab} & " + " & ".join(c) + " \\\\")
        if task != TASKS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "blind_spot_by_k.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/blind_spot_by_k.tex")


if __name__ == "__main__":
    main()
    window_table()
