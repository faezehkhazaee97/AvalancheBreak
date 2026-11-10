#!/usr/bin/env python3
"""Two follow-up analyses for the review response.

1. Wording test (tasks/common/wording.py). Does the post-marker blind spot depend on calling the
   bars "structural boundaries"? For running_xor and running_sum_local at Qwen2.5-32B,
   Qwen3.8-27B and Gemma-4-31B, the per-position comparison of scripts/analyze_per_position.py
   (post-marker and within-segment positions, each against the no-marker run of the same
   prompt variant at the same indices; N <= 64, k in {2,4,8,16}; index scoring) is repeated for
   the original, neutral and across wordings.
   Output: results_clean/_summary/wording_per_position.csv

2. Head-count dose response (scripts/run_head_ablation_v2.py with --top 10/50/200). At
   Qwen2.5-32B, random_permutation spaced letters and digits, A = 8, N = 64, k = 8: the share of
   the marker gain removed by blocking the top-n marker-attending heads, and by the random-head
   and sham-position controls, with paired 95% intervals on the drop.
   Output: results_clean/_summary/head_dose.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load  # noqa: E402
from analyze_aligned_scoring import summarize, index_vector  # noqa: E402

MODELS = [("Qwen2.5 32B", "Qwen2.5-32B-Instruct"), ("Qwen3.8 27B", "Qwen3.8-27B"), ("Gemma 4 31B", "gemma-4-31B-it")]
TASKS = ["running_xor", "running_sum_local"]
VARIANTS = [("original", ""), ("neutral", "_neutral"), ("across", "_across")]


def wording():
    rows = []
    for task in TASKS:
        for label, tag in MODELS:
            for v, suf in VARIANTS:
                d = load(task + suf, tag)
                if d is None:
                    continue
                r = summarize(d, index_vector)
                rows.append(dict(task=task, model=label, wording=v, **{k: round(float(x), 4) for k, x in r.items()}))
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results_clean" / "_summary" / "wording_per_position.csv", index=False)
    print(out.pivot_table(index=["task", "model"], columns="wording", values=["post_gain", "within_gain"])
          .round(2).to_string())
    print(out.pivot_table(index=["task", "model"], columns="wording", values=["post", "base"]).round(2).to_string())
    return out


def dose():
    src = {10: "Qwen2.5-32B-Instruct.csv", 50: "Qwen2.5-32B-Instruct_top50.csv", 200: "Qwen2.5-32B-Instruct_top200.csv"}
    rows = []
    for n, f in src.items():
        d = pd.read_csv(REPO / "results" / "ablation_v2" / f)
        for task in ["random_permutation_spaced", "random_permutation_digits"]:
            x = d[(d.Task == task) & (d.Seq_Length == 64)]
            base = x[x.Condition == "base"].TAR.mean()
            b = x[(x.Interval == 8)].pivot_table(index="Trial", columns="Condition", values="TAR")
            gain = b["bounds"].mean() - base
            for cond in ["ablate_top", "ctrl_random", "ctrl_sham"]:
                drop = b["bounds"] - b[cond]
                rows.append(dict(task=task, heads=n, condition=cond, base=round(base, 3),
                                 bounds=round(b["bounds"].mean(), 3), blocked=round(b[cond].mean(), 3),
                                 drop=round(drop.mean(), 3), ci=round(1.96 * drop.sem(), 3),
                                 share=round(drop.mean() / gain, 3) if gain > 0 else np.nan, trials=len(b)))
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results_clean" / "_summary" / "head_dose.csv", index=False)
    print(out.to_string(index=False))
    return out


def fmt(x):
    return f"${x:+.2f}$"


def write_tables(w, d):
    tdir = REPO / "paper_tmlr" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{tabular}{ll ccc ccc}", "\\toprule",
             "& & \\multicolumn{3}{c}{Post-marker} & \\multicolumn{3}{c}{Within-segment} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "Task & Model & Original & Neutral & Across & Original & Neutral & Across \\\\", "\\midrule"]
    for task in TASKS:
        for label, _ in MODELS:
            x = w[(w.task == task) & (w.model == label)].set_index("wording")
            cells = [fmt(x.loc[v, "post_gain"]) for v, _ in VARIANTS] + [fmt(x.loc[v, "within_gain"]) for v, _ in VARIANTS]
            lines.append(f"\\task{{{task.replace('_', chr(92) + '_')}}} & {label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "wording.tex").write_text("\n".join(lines) + "\n")
    lines = ["\\begin{tabular}{ll ccc}", "\\toprule",
             "Format & Heads blocked & Top $n$ & Random $n$ & Sham \\\\", "\\midrule"]
    for task, lab in [("random_permutation_spaced", "Spaced letters"), ("random_permutation_digits", "Digits")]:
        for n in (10, 50, 200):
            x = d[(d.task == task) & (d.heads == n)].set_index("condition")
            cells = [f"${int(round(100 * x.loc[c, 'share']))}\\%$ (${x.loc[c, 'drop']:+.3f} \\pm {x.loc[c, 'ci']:.3f}$)"
                     for c in ("ablate_top", "ctrl_random", "ctrl_sham")]
            lines.append(f"{lab if n == 10 else ''} & {n} & " + " & ".join(cells) + " \\\\")
        if task.endswith("spaced"):
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "head_dose.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/wording.tex and head_dose.tex")


if __name__ == "__main__":
    write_tables(wording(), dose())
