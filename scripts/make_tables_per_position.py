#!/usr/bin/env python3
"""Write the paper's per-position appendix tables as LaTeX, straight from the analysis CSVs.

  paper_tmlr/tables/perpos.tex        : block means per model (Table tab:perpos)
  paper_tmlr/tables/perpos_tasks.tex  : the same comparison per task (letters, spaced letters,
                                        digits, dependency blocks), replacing the pointer to
                                        results_clean/_summary/per_position_values.csv
  paper_tmlr/tables/distance.tex      : accuracy by distance after the marker, k = 16
                                        (tab:distance), with sliding_window_max also on its own
  paper_tmlr/tables/aligned.tex       : index vs alignment-scored per-position gains (tab:aligned),
                                        with the alignment-sensitive share of the within-segment gain

Also prints the post-marker vs within-segment comparison on the tokenization controls
(spaced letters, digits) that the main text cites.

Run scripts/analyze_per_position.py (and scripts/analyze_aligned_scoring.py) first.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results_clean" / "_summary"
OUT = REPO / "paper_tmlr" / "tables"
MODELS = ["Qwen2.5 7B", "Qwen2.5 14B", "Qwen2.5 32B", "Mistral 7B", "Phi-3.5 mini",
          "Qwen3.8 27B", "Gemma 4 31B", "Gemini 3.5 Flash", "Gemini 3.6 Flash"]
OPEN = MODELS[:7]
STRONG = ["Qwen2.5 14B", "Qwen2.5 32B", "Qwen3.8 27B", "Gemma 4 31B"]
BLOCKS = [("Position-local, letters", "Position-local tasks, letters"),
          ("Position-local, spaced letters", "Position-local tasks, spaced letters"),
          ("Position-local, digits", "Position-local tasks, digits"),
          ("Dependency, as is", "Dependency tasks"),
          ("Dependency severed (reset)", "Dependency tasks, reset variants")]


def f(x):
    return f"{x:.2f}"


def signed(x):
    x = round(x, 2) + 0.0  # avoid "-0.00"
    return "$0.00$" if x == 0 else f"${'+' if x > 0 else '-'}{abs(x):.2f}$"


def task_name(t):
    for suf in ("_spaced", "_digits"):
        if t.endswith(suf):
            t = t[: -len(suf)]
    return r"\task{" + t.replace("_", r"\_") + "}"


def perpos():
    v = pd.read_csv(SUMMARY / "per_position_values.csv")
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{Within a segment} & \multicolumn{3}{c}{Right after a marker} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Model & No markers & Markers & $\Delta$ & No markers & Markers & $\Delta$ \\", r"\midrule"]
    for block, title in BLOCKS:
        s = v[v.block == block]
        if s.empty:
            continue
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}} \\")
        for m in MODELS:
            x = s[s.model == m]
            if x.empty:
                continue
            bw, w, bp, p = x.base_at_within.mean(), x.within.mean(), x.base_at_post.mean(), x.post.mean()
            lines.append(f"{m} & {f(bw)} & {f(w)} & {signed(w - bw)} & {f(bp)} & {f(p)} & {signed(p - bp)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "perpos.tex").write_text("\n".join(lines) + "\n")


def perpos_tasks():
    """Per-task version of tab:perpos (letters, spaced letters, digits, dependency blocks)."""
    v = pd.read_csv(SUMMARY / "per_position_values.csv")
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"& & \multicolumn{3}{c}{Within a segment} & \multicolumn{3}{c}{Right after a marker} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Task & Model & No markers & Markers & $\Delta$ & No markers & Markers & $\Delta$ \\", r"\midrule"]
    for block, title in BLOCKS[:4]:
        s = v[v.block == block]
        if s.empty:
            continue
        if lines[-1] != r"\midrule":
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{title}}}}} \\")
        for task in s.task.drop_duplicates():
            first = True
            for m in MODELS:
                x = s[(s.task == task) & (s.model == m)]
                if x.empty:
                    continue
                r = x.iloc[0]
                bw, w, bp, p = r.base_at_within, r.within, r.base_at_post, r.post
                lines.append(f"{task_name(task) if first else ''} & {m} & {f(bw)} & {f(w)} & {signed(w - bw)} & "
                             f"{f(bp)} & {f(p)} & {signed(p - bp)} \\\\")
                first = False
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "perpos_tasks.tex").write_text("\n".join(lines) + "\n")


def controls_report():
    """Post-marker vs within-segment positions on the tokenization controls (printed only)."""
    v = pd.read_csv(SUMMARY / "per_position_values.csv")
    v["dpost"] = v.post - v.base_at_post
    v["dwithin"] = v.within - v.base_at_within
    print("\nPost-marker vs within-segment on the tokenization controls (N <= 64, k in {2,4,8,16})")
    for block in ["Position-local, spaced letters", "Position-local, digits"]:
        s = v[v.block == block]
        print(f"\n  {block}")
        for lab, models in [("open models", OPEN), ("open models + Gemini 3.6", OPEN + ["Gemini 3.6 Flash"])]:
            t = s[s.model.isin(models)]
            print(f"    task x model cells ({lab}, n = {len(t)}): post > within on raw marked accuracy in "
                  f"{int((t.post > t.within).sum())}; on the baseline-matched change in {int((t.dpost > t.dwithin).sum())}")
            m = t.groupby("model")[["post", "within", "dpost", "dwithin"]].mean().reindex([x for x in models if x in set(t.model)])
            print(f"    model rows of tab:perpos ({lab}, n = {len(m)}): post > within raw in {int((m.post > m.within).sum())}, "
                  f"change in {int((m.dpost > m.dwithin).sum())}")
            raw, chg = (m.post - m.within), (m.dpost - m.dwithin)
            print(f"    largest |post - within| over model rows: raw {raw.abs().max():.3f} ({raw.abs().idxmax()}), "
                  f"change {chg.abs().max():.3f} ({chg.abs().idxmax()})")
        q = s[s.model == "Qwen2.5 7B"][["task", "within", "post", "dwithin", "dpost"]]
        qm = q[["within", "post", "dwithin", "dpost"]].mean()
        print(f"    Qwen2.5 7B block mean: within {qm.within:.3f}, post {qm.post:.3f} (raw post - within "
              f"{qm.post - qm.within:+.3f}); change within {qm.dwithin:+.3f}, post {qm.dpost:+.3f} "
              f"(post - within {qm.dpost - qm.dwithin:+.3f})")
        print(q.round(3).to_string(index=False))


def distance():
    d = pd.read_csv(SUMMARY / "per_position_distance.csv")
    d = d[d.k == 16]
    bins = [("0", [0]), ("1", [1]), ("2--3", [2, 3]), ("4--7", [4, 5, 6, 7]), ("8--15", list(range(8, 16)))]
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"& & \multicolumn{5}{c}{Positions after the marker} \\", r"\cmidrule(lr){3-7}",
             "Tasks & Model & " + " & ".join(b for b, _ in bins) + r" \\", r"\midrule"]
    swm = d[d.task == "sliding_window_max"].assign(block="sliding_window_max")
    for block, title in [("Position-local, letters", "Position-local, letters"),
                         ("Position-local, spaced letters", "Position-local, spaced"),
                         ("Position-local, digits", "Position-local, digits"),
                         ("Dependency, as is", "Dependency"),
                         ("sliding_window_max", r"\task{sliding\_window\_max}")]:
        s = swm if block == "sliding_window_max" else d[d.block == block]
        if s.empty:
            continue
        first = True
        for m in MODELS:
            x = s[s.model == m]
            if x.empty:
                continue
            cells = " & ".join(f(x[x.d.isin(dd)].acc.mean()) for _, dd in bins)
            lines.append(f"{title if first else ''} & {m} & {cells} \\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (OUT / "distance.tex").write_text("\n".join(lines) + "\n")
    s = swm[swm.model.isin(STRONG)]
    print("\nsliding_window_max, k = 16, accuracy by distance after the marker:")
    print(s.pivot_table(index="model", columns="d", values="acc").reindex(columns=range(16)).round(2).to_string())


def share_cell(idx, ali):
    """Alignment-sensitive share of a within-segment gain, shown only where the index-scored gain is
    at least +0.02 and the aligned gain is not negative. Otherwise the ratio leaves [0, 1] and
    carries no meaning: the 7B-class cells produced shares of 152% and 270% this way."""
    if round(idx, 2) < 0.02 or ali < 0:
        return "--"
    s = (idx - ali) / idx
    return f"${'-' if s < 0 else ''}{abs(100 * s):.0f}\\%$"


def aligned():
    """Appendix table: per-position gains under index-aligned vs edit-distance-aligned scoring
    (results_clean/_summary/aligned_scoring.csv from scripts/analyze_aligned_scoring.py), plus the
    share of the index-scored within-segment gain that alignment removes, (index - aligned) / index,
    where the index-scored within-segment gain is positive (dash otherwise)."""
    fpath = SUMMARY / "aligned_scoring.csv"
    if not fpath.exists():
        return
    a = pd.read_csv(fpath).pivot_table(index=["task", "model"], columns="scorer", values=["post_gain", "within_gain"])
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"& & \multicolumn{2}{c}{Right after a marker} & \multicolumn{3}{c}{Within a segment} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
             r"Task & Model & Index & Aligned & Index & Aligned & Share \\", r"\midrule"]
    shares = []
    for task in ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]:
        first = True
        for m in MODELS:
            if (task, m) not in a.index:
                continue
            r = a.loc[(task, m)]
            wi, wa = r[("within_gain", "index")], r[("within_gain", "aligned")]
            name = r"\task{" + task.replace("_", r"\_") + "}" if first else ""
            lines.append(f"{name} & {m} & {signed(r[('post_gain', 'index')])} & {signed(r[('post_gain', 'aligned')])} & "
                         f"{signed(wi)} & {signed(wa)} & {share_cell(wi, wa)} \\\\")
            if m in STRONG:
                shares.append(dict(task=task, model=m, index=wi, aligned=wa, share=(wi - wa) / wi if round(wi, 2) >= 0.02 else np.nan))
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (OUT / "aligned.tex").write_text("\n".join(lines) + "\n")
    s = pd.DataFrame(shares)
    pos = s.dropna()
    print(f"\nAlignment-sensitive share of the within-segment gain, four stronger models: {len(pos)} of {len(s)} "
          f"dependency cells have a positive index-scored gain; median {pos.share.median():.3f}, "
          f"range {pos.share.min():.3f} to {pos.share.max():.3f}")
    print(s.round(3).to_string(index=False))
    print(f"Wrote {OUT / 'aligned.tex'}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    perpos()
    perpos_tasks()
    distance()
    print(f"Wrote {OUT / 'perpos.tex'}, {OUT / 'perpos_tasks.tex'} and {OUT / 'distance.tex'}")
    aligned()
    controls_report()
