#!/usr/bin/env python3
"""Worked-example wording (tasks/common/wording.py, make_worked): does showing the model two solved
examples with bars, one annotated at the position right after a bar, remove the post-marker blind
spot? For each dependency task and model with a *_worked run, the per-position comparison of
scripts/analyze_per_position.py (post-marker and within-segment positions, each against the
no-marker run of the same wording at the same indices; N <= 64, k in {2,4,8,16}; index scoring)
is given for the original wording and the worked one.

Outputs: results_clean/_summary/worked_per_position.csv, paper_tmlr/tables/worked.tex
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS as ALL_MODELS  # noqa: E402
from analyze_aligned_scoring import summarize, index_vector  # noqa: E402

TASKS = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]
VARIANTS = [("original", ""), ("worked", "_worked")]


def main():
    rows = []
    for task in TASKS:
        for label, tag in ALL_MODELS:
            if tag is None or str(tag).startswith("gemini") or load(task + "_worked", tag) is None:
                continue
            for v, suf in VARIANTS:
                d = load(task + suf, tag)
                r = summarize(d, index_vector)
                rows.append(dict(task=task, model=label, wording=v, n=len(d),
                                 **{k: round(float(x), 4) for k, x in r.items()}))
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results_clean" / "_summary" / "worked_per_position.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.pivot_table(index=["task", "model"], columns="wording", values=["base", "post", "post_gain", "within_gain"])
          .round(3).to_string())

    def f(x):
        return f"${x:+.2f}$"
    lines = ["\\begin{tabular}{ll cc cc cc}", "\\toprule",
             "& & \\multicolumn{2}{c}{No-marker acc.} & \\multicolumn{2}{c}{Post-marker} & \\multicolumn{2}{c}{Within-segment} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
             "Task & Model & Original & Worked & Original & Worked & Original & Worked \\\\", "\\midrule"]
    for task in TASKS:
        first = True
        for label, _ in ALL_MODELS:
            x = out[(out.task == task) & (out.model == label)].set_index("wording")
            if len(x) < 2:
                continue
            cells = ([f"{x.loc[v, 'base']:.2f}" for v, _ in VARIANTS] + [f(x.loc[v, "post_gain"]) for v, _ in VARIANTS]
                     + [f(x.loc[v, "within_gain"]) for v, _ in VARIANTS])
            lines.append(f"{'\\task{' + task.replace('_', chr(92) + '_') + '}' if first else ''} & {label} & "
                         + " & ".join(cells) + " \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "worked.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/worked.tex")


if __name__ == "__main__":
    main()
