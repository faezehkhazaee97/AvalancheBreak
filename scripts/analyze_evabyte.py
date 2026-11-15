#!/usr/bin/env python3
"""Byte-level model EvaByte-SFT (one token per character and per marker) on two dependency tasks and
two position-local digit tasks, analyzed as for the other models: no-marker accuracy (length-matched),
pooled Delta and z (N <= 64, k in {1, 2, 4, 8, 16}), and the change at post-marker and within-segment
positions against the no-marker run at the same indices (k in {2, 4, 8, 16}). Also the same numbers
for Qwen2.5-7B, the closest model in size, for comparison.
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
MODELS = [("EvaByte-SFT", "EvaByte-SFT"), ("Qwen2.5 7B", "Qwen2.5-7B-Instruct")]


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
        for label, tag in MODELS:
            f = REPO / "results_clean" / task / "grid" / f"{tag}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, dtype={"Ground_Truth": str, "LLM_Output": str, "Raw_Model_Output": str}, keep_default_na=False)
            r = pooled_delta(d, lengths=[4, 8, 16, 32, 64])
            pp = per_position(d)
            rows.append(dict(task=task, model=label, base=r["base"], bounds=r["bounds"], delta=r["delta"], z=r["z"],
                             post=pp["post"], within=pp["within"], n=len(d)))
    t = pd.DataFrame(rows)
    t.to_csv(REPO / "results_clean" / "_summary" / "evabyte.csv", index=False)
    print(t.round(3).to_string(index=False))
    lines = ["\\begin{tabular}{ll cc cc}", "\\toprule",
             "Task & Model & No markers & Pooled $\\Delta$ ($z$) & Post-marker & Within segment \\\\", "\\midrule"]
    for task, lab in TASKS:
        first = True
        for r in t[t.task == task].itertuples():
            lines.append(f"{lab if first else ''} & {r.model} & {r.base:.2f} & ${r.delta:+.2f}$ ({abs(r.z):.1f}) & ${r.post:+.2f}$ & ${r.within:+.2f}$ \\\\")
            first = False
        lines.append("\\addlinespace")
    lines[-1:] = ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "evabyte.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/evabyte.tex")


if __name__ == "__main__":
    main()
