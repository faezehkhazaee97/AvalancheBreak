#!/usr/bin/env python3
"""Analysis of scripts/run_base_model.py: a base checkpoint against its instruction-tuned version,
both prompted the same way (no chat template, three solved examples). Per model and task family:
no-marker accuracy, the pooled marker effect (every k > 0 against k = 0 at the same lengths), and
the per-position comparison of scripts/analyze_per_position.py (post-marker and within-segment
positions against the no-marker run at the same indices; k in {2,4,8,16}; index scoring).

Outputs: results_clean/_summary/base_model.csv, paper_tmlr/tables/base_model.tex
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_aligned_scoring import summarize, index_vector  # noqa: E402

TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}
MODELS = [("Qwen2.5-7B (base)", "Qwen2.5-7B"), ("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct")]
FAM = {"Digits": ["cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"],
       "Dependency": ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]}


def main():
    rows = []
    for label, tag in MODELS:
        f = REPO / "results" / "base_model" / f"{tag}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f, dtype=TEXT, keep_default_na=False)
        p = p.assign(N=p.Seq_Length.astype(int), k=p.Interval.astype(int), TAR=p.TAR.astype(float))
        for fam, tasks in FAM.items():
            for task in tasks:
                x = p[p.Task == task]
                if x.empty:
                    continue
                base = x[x.k == 0].groupby("N").TAR.mean()
                m = x[x.k > 0]
                delta = (m.TAR - m.N.map(base)).mean()          # length-matched pooled effect
                s = summarize(x, index_vector)
                rows.append(dict(model=label, family=fam, task=task, n=len(x), base=base.mean(), delta=delta,
                                 post=s["post"], post_gain=s["post_gain"], within_gain=s["within_gain"]))
    o = pd.DataFrame(rows).round(4)
    o.to_csv(REPO / "results_clean" / "_summary" / "base_model.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.to_string(index=False))
    s = o.groupby(["model", "family"])[["base", "delta", "post", "post_gain", "within_gain"]].mean()
    print(s.round(3).to_string())

    lines = ["\\begin{tabular}{ll cccc}", "\\toprule",
             "Model & Format & No-marker acc. & $\\Delta$ & Post-marker & Within-segment \\\\", "\\midrule"]
    for label, _ in MODELS:
        first = True
        for fam in FAM:
            if (label, fam) not in s.index:
                continue
            r = s.loc[(label, fam)]
            lines.append(f"{label if first else ''} & {fam} & {r.base:.2f} & ${r.delta:+.2f}$ & ${r.post_gain:+.2f}$ & "
                         f"${r.within_gain:+.2f}$ \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "base_model.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/base_model.tex")


if __name__ == "__main__":
    main()
