#!/usr/bin/env python3
"""Finetuned separator (reviewer question): does a LoRA adapter trained to carry a dependency
across a marker remove the blind spot, and does it transfer to tasks it was not trained on?

scripts/train_lora_separator.py produces two merged checkpoints of Qwen2.5-7B-Instruct: one trained
on running_xor and running_sum_local with marked targets at every spacing (lorasep), and a control
trained on the same examples at k = 0 only (lorabase). scripts/eval_lora.sbatch evaluates them with
the usual runners on running_xor (trained task, unseen grid inputs), copy_previous and
sliding_window_max (dependency tasks not in training), and random_permutation_digits (a
position-local digit task). For each checkpoint and task: no-marker accuracy, the pooled
length-matched marker effect, and the per-position comparison (post-marker and within-segment
positions against the no-marker run at the same indices; k in {2,4,8,16}; index scoring).

Outputs: results_clean/_summary/lora.csv, paper_tmlr/tables/lora.tex
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_naturalistic import row  # noqa: E402

TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}
MODELS = [("Qwen2.5 7B (as released)", "Qwen2.5-7B-Instruct"),
          ("+ LoRA, k = 0 only (control)", "Qwen2.5-7B-Instruct-lorabase"),
          ("+ LoRA, marked targets", "Qwen2.5-7B-Instruct-lorasep")]
TASKS = [("running_xor", "Running XOR (trained)"), ("copy_previous", "Copy previous (held out)"),
         ("sliding_window_max", "Sliding-window max (held out)"), ("random_permutation_digits", "Random permutation, digits (held out)")]


def load_exact(task, tag):
    f = REPO / "results_clean" / task / "grid" / f"{tag}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, dtype=TEXT, keep_default_na=False)
    axis = next((c for c in ("Alphabet_Size", "Modulus") if c in d.columns), None)
    return d.drop_duplicates(subset=["Method", "Interval", "Seq_Length", "Trial"] + ([axis] if axis else []))


def main():
    rows = []
    for task, _ in TASKS:
        for label, tag in MODELS:
            d = load_exact(task, tag)
            if d is not None:
                rows.append(row(task, label, d))
    out = pd.DataFrame(rows).round(4)
    if out.empty:
        print("no results yet"); return
    out.to_csv(REPO / "results_clean" / "_summary" / "lora.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))

    lines = ["\\begin{tabular}{ll cccc}", "\\toprule",
             "Task & Model & No-marker acc. & $\\Delta$ & Post-marker & Within-segment \\\\", "\\midrule"]
    for task, tname in TASKS:
        first = True
        for label, _ in MODELS:
            x = out[(out.task == task) & (out.model == label)]
            if x.empty:
                continue
            r = x.iloc[0]
            lines.append(f"{tname if first else ''} & {label} & {r.base:.2f} & ${r.delta:+.2f}$ & ${r.post_gain:+.2f}$ & ${r.within_gain:+.2f}$ \\\\")
            first = False
        if not first:
            lines.append("\\addlinespace")
    if lines[-1] == "\\addlinespace":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "lora.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/lora.tex")


if __name__ == "__main__":
    main()
