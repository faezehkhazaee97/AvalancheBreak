#!/usr/bin/env python3
"""Byte-level model EvaByte-SFT (one token per character and per marker) on two dependency tasks and
two position-local digit tasks, analyzed as for the other models: no-marker accuracy (length-matched),
pooled Delta and z (N <= 64, k in {1, 2, 4, 8, 16}), and the change at post-marker and within-segment
positions against the no-marker run at the same indices (k in {2, 4, 8, 16}). Also the same numbers
for Qwen2.5-7B, the closest model in size, for comparison.

Two EvaByte runs are reported. The zero-shot run (results_clean/<task>/grid/EvaByte-SFT.csv) is the
one Appendix L describes, where the model writes its input back instead of applying the rule. The
3-shot run (results/evabyte_fewshot/<task>.csv, scripts/run_evabyte_fewshot.py) gives it the three
solved examples the base checkpoints get (scripts/run_base_model.py), the same grid and scoring, so
the only change is the demonstrations. The "Copies input" column is the share of outputs equal to
the input with the markers removed -- the adherence failure the appendix quantifies -- and is what
says whether the few-shot numbers test the markers at all.

Outputs: results_clean/_summary/evabyte.csv, paper_tmlr/tables/evabyte.tex
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.pooled import pooled_delta  # noqa: E402

TASKS = [("running_xor", "\\task{running\\_xor}"), ("copy_previous", "\\task{copy\\_previous}"),
         ("random_permutation_digits", "\\task{random\\_permutation}, digits"), ("cyclic_shift_digits", "\\task{cyclic\\_shift}, digits")]
# label, path relative to the repo with {task} filled in
MODELS = [("EvaByte-SFT, zero-shot", "results_clean/{task}/grid/EvaByte-SFT.csv"),
          ("EvaByte-SFT, 3-shot", "results/evabyte_fewshot/{task}.csv"),
          ("Qwen2.5 7B", "results_clean/{task}/grid/Qwen2.5-7B-Instruct.csv")]


def copies_input(d):
    """Share of rows whose scored output is exactly the input with the markers removed."""
    return float((d.LLM_Output == d.Input_Given.str.replace("|", "", regex=False)).mean())


def per_position(d):
    d = d[d.Seq_Length.astype(int) <= 64]
    ax = next((c for c in ("Alphabet_Size", "Modulus") if c in d.columns), None)
    base = {}
    for _, r in d[d.Interval.astype(int) == 0].iterrows():
        t, p = str(r.Ground_Truth), str(r.LLM_Output)
        for i, c in enumerate(t):
            base.setdefault((int(r.Seq_Length), r[ax] if ax else 0, i), []).append(float(i < len(p) and p[i] == c))
    acc = {"post": [], "within": []}
    for _, r in d[d.Interval.astype(int).isin([2, 4, 8, 16])].iterrows():
        k, N = int(r.Interval), int(r.Seq_Length)
        t, p = str(r.Ground_Truth), str(r.LLM_Output)
        for i in range(1, len(t)):
            b = base.get((N, r[ax] if ax else 0, i))
            if b is None:
                continue
            acc["post" if i % k == 0 else "within"].append(float(i < len(p) and p[i] == t[i]) - np.mean(b))
    return {k: float(np.mean(v)) if v else np.nan for k, v in acc.items()}


def main():
    rows = []
    for task, lab in TASKS:
        for label, path in MODELS:
            f = REPO / path.format(task=task)
            if not f.exists():
                print(f"missing: {f.relative_to(REPO)}")
                continue
            d = pd.read_csv(f, dtype={"Input_Given": str, "Ground_Truth": str, "LLM_Output": str,
                                      "Raw_Model_Output": str}, keep_default_na=False)
            r = pooled_delta(d, lengths=[4, 8, 16, 32, 64])
            pp = per_position(d)
            d64 = d[d.Seq_Length.astype(int) <= 64]
            rows.append(dict(task=task, model=label, base=r["base"], bounds=r["bounds"], delta=r["delta"], z=r["z"],
                             post=pp["post"], within=pp["within"],
                             copies=copies_input(d64), n=len(d), n64=len(d64)))
    t = pd.DataFrame(rows)
    t.to_csv(REPO / "results_clean" / "_summary" / "evabyte.csv", index=False)
    print(t.round(3).to_string(index=False))
    print("\ncopies input, pooled over the four tasks (row-weighted, the figure quoted in the appendix):")
    print(t.groupby("model").apply(lambda g: (g.copies * g.n64).sum() / g.n64.sum(), include_groups=False)
          .round(3).to_string())
    lines = ["\\begin{tabular}{ll ccc cc}", "\\toprule",
             "Task & Model & No markers & Copies input & Pooled $\\Delta$ ($z$) & Post-marker & Within segment \\\\",
             "\\midrule"]
    for task, lab in TASKS:
        first = True
        for r in t[t.task == task].itertuples():
            lines.append(f"{lab if first else ''} & {r.model} & {r.base:.2f} & {r.copies:.2f} & "
                         f"${r.delta:+.2f}$ ({abs(r.z):.1f}) & ${r.post:+.2f}$ & ${r.within:+.2f}$ \\\\")
            first = False
        lines.append("\\addlinespace")
    lines[-1:] = ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "evabyte.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/evabyte.tex")


if __name__ == "__main__":
    main()
