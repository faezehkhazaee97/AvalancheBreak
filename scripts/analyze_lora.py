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
# (base model, row label, checkpoint tag); labels repeat across base models, so rows are keyed by tag
MODELS = [("Qwen2.5 7B", "as released", "Qwen2.5-7B-Instruct"),
          ("Qwen2.5 7B", "+ LoRA, $k = 0$ only (control)", "Qwen2.5-7B-Instruct-lorabase"),
          ("Qwen2.5 7B", "+ LoRA, $k = 0$ only, matched steps", "Qwen2.5-7B-Instruct-lorabase-matched"),
          ("Qwen2.5 7B", "+ LoRA, marked targets", "Qwen2.5-7B-Instruct-lorasep"),
          ("Qwen2.5 32B", "as released", "Qwen2.5-32B-Instruct"),
          ("Qwen2.5 32B", "+ LoRA, $k = 0$ only (control)", "Qwen2.5-32B-Instruct-lorabase"),
          ("Qwen2.5 32B", "+ LoRA, marked targets", "Qwen2.5-32B-Instruct-lorasep")]
TASKS = [("running_xor", "XOR (trained)"), ("running_sum_local", "Sum (trained)"), ("copy_previous", "Copy previous (held out)"),
         ("sliding_window_max", "Window max (held out)"), ("random_permutation_digits", "Permutation, digits (held out)")]


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
        for base, label, tag in MODELS:
            d = load_exact(task, tag)
            if d is not None:
                r = row(task, label, d); r.update(base_model=base, tag=tag)
                rows.append(r)
    out = pd.DataFrame(rows).round(4)
    if out.empty:
        print("no results yet"); return
    out.to_csv(REPO / "results_clean" / "_summary" / "lora.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[["base_model", "model", "task", "n", "base", "delta", "post", "post_gain", "within_gain"]].to_string(index=False))

    def cell(r):
        return f"{r.base:.2f} & ${r.post_gain:+.2f}$ & ${r.within_gain:+.2f}$"
    tasks = [t for t, _ in TASKS if (out.task == t).any()]
    head = " & ".join("\\multicolumn{3}{c}{" + dict(TASKS)[t] + "}" for t in tasks)
    cmid = "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(tasks)))
    sub = " & ".join(["Base & Post & Within"] * len(tasks))
    lines = ["\\begin{tabular}{l" + " ccc" * len(tasks) + "}", "\\toprule", "& " + head + " \\\\", cmid,
             "Checkpoint & " + sub + " \\\\", "\\midrule"]
    for bi, base in enumerate(dict.fromkeys(b for b, _, _ in MODELS)):
        if bi:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{1 + 3 * len(tasks)}}}{{l}}{{\\emph{{{base}}}}} \\\\")
        for b, label, tag in MODELS:
            if b != base:
                continue
            cells = []
            for t in tasks:
                x = out[(out.task == t) & (out.tag == tag)]
                cells.append(cell(x.iloc[0]) if len(x) else "-- & -- & --")
            if all(c.startswith("--") for c in cells):
                continue
            lines.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "lora.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/lora.tex")


if __name__ == "__main__":
    main()
