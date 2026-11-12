#!/usr/bin/env python3
"""Appendix tables of pooled effects, from results_clean/_summary/pooled_all.csv
(run scripts/summarize_pooled.py first). Writes paper_tmlr/tables/{reset,gemini,external}.tex.
All effects are length-matched (tasks/common/pooled.py)."""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from summarize_pooled import grid, MODELS  # noqa: E402

OUT = REPO / "paper_tmlr" / "tables"
P = pd.read_csv(REPO / "results_clean" / "_summary" / "pooled_all.csv")


def tt(t):
    return r"\task{" + t.replace("_", r"\_") + "}"


def dz(r, signed_z=False):
    d = 0.0 if abs(r.delta) < 0.0005 else r.delta   # avoid printing "-0.000"
    s = f"{'+' if d > 0 else ('$-$' if d < 0 else '')}{abs(d):.3f}"
    return f"{s} ({abs(r.z):.1f})"


def get(family, task, model):
    x = P[(P.family == family) & (P.task == task) & (P.model == model)]
    return None if x.empty else x.iloc[0]


def reset_table():
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"& & \multicolumn{2}{c}{Original (dependency intact)} & \multicolumn{2}{c}{Reset (dependency severed)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Task & Model & Base$\rightarrow$Bounds & $\Delta$ ($|z|$) & Base$\rightarrow$Bounds & $\Delta$ ($|z|$) \\",
             r"\midrule"]
    for t in ["running_xor", "running_sum_local", "sliding_window_max", "copy_previous"]:
        first = True
        for m in ["Qwen2.5 7B", "Qwen2.5 32B", "Qwen3.8 27B", "Gemma 4 31B"]:
            o, r = get("dependency", t, m), get("reset", t, m)
            if o is None or r is None:
                continue
            lines.append(f"{tt(t) if first else ''} & {m.replace('Qwen2.5 ', '')} & {o.base:.3f}$\\rightarrow${o.bounds:.3f} & "
                         f"{dz(o)} & {r.base:.3f}$\\rightarrow${r.bounds:.3f} & {dz(r)} \\\\")
            first = False
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "reset.tex").write_text("\n".join(lines) + "\n")


def gemini_table():
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Task & Model & Base acc. & Bounds acc. & $\Delta$ ($|z|$) \\", r"\midrule"]
    for t in ["cyclic_shift", "random_permutation", "conditional_shift", "running_xor", "running_sum_local", "copy_previous",
              "sliding_window_max"]:
        first = True
        for lab, model in [("Gemini 3.5", "Gemini 3.5 Flash"), ("Gemini 3.6", "Gemini 3.6 Flash")]:
            r = get("gemini_table", t, model)
            if r is None:       # sliding_window_max was not run on Gemini 3.5
                continue
            lines.append(f"{tt(t) if first else ''} & {lab} & {r.base:.3f} & {r.bounds:.3f} & {dz(r)} \\\\")
            first = False
    for t in ["cyclic_shift_spaced", "random_permutation_spaced", "conditional_shift_spaced", "cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"]:
        r6 = get("gemini_table", t, "Gemini 3.6 Flash")
        if r6 is not None:
            lines.append(f"{tt(t)} & Gemini 3.6 & {r6.base:.3f} & {r6.bounds:.3f} & {dz(r6)} \\\\")
    lines.append(r"\midrule")
    for t, lab in [("clrs_minimum", "CLRS minimum"), ("clrs_kadane", "CLRS Kadane")]:
        first = True
        for m in ["Qwen2.5 7B", "Qwen2.5 32B", "Gemini 3.5 Flash"]:
            r = get("external", t, m)
            if r is None:
                continue
            name = m.replace("Qwen2.5 ", "Qwen2.5-").replace("Gemini 3.5 Flash", "Gemini 3.5")
            lines.append(f"{lab if first else ''} & {name} & {r.base:.3f} & {r.bounds:.3f} & {dz(r)} \\\\")
            first = False
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "gemini.tex").write_text("\n".join(lines) + "\n")


def external_table():
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{Qwen2.5-7B} & \multicolumn{3}{c}{Qwen2.5-32B} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Task & Base & Best-$k$ & $\Delta$ ($|z|$) & Base & Best-$k$ & $\Delta$ ($|z|$) \\", r"\midrule"]
    for t, lab in [("vowel_tagging", "Vowel/consonant tagging"), ("procbench_substitute", "ProcBench-Substitute"),
                   ("string_copy", "String copying")]:
        cells = [lab]
        for m in ["Qwen2.5 7B", "Qwen2.5 32B"]:
            r = get("external", t, m)
            if t == "vowel_tagging":
                d = pd.read_csv(REPO / "results" / MODELS[m] / "vowel_tagging_boundary_sweep.csv")
            else:
                d = grid(t, MODELS[m])
            d = d[(d.Interval != 3) & (d.Interval > 0)]
            best = d.groupby("Interval").TAR.mean().max()
            cells += [f"{r.base_unweighted:.3f}", f"{best:.3f}", dz(r)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "external.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    reset_table()
    gemini_table()
    external_table()
    print(f"Wrote {OUT}/reset.tex, gemini.tex, external.tex")
