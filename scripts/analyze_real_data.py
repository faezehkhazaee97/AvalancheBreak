#!/usr/bin/env python3
"""Real sequences (tasks/dna_* and tasks/text_*): the three effects on windows of human promoter DNA
and of WikiText-2 letters, with a position-local, a one-step-dependency and a prefix-state task per
source, each in the contiguous and the spaced-character format. Per task, format and model:
no-marker accuracy, the pooled length-matched marker effect (k = 3 excluded, as in the main grid),
and the per-position comparison (post-marker and within-segment positions against the no-marker
run at the same indices; k in {2,4,8,16}; index scoring). Also the spacing-alone effect: the spaced
no-marker run against the contiguous no-marker run.

Outputs: results_clean/_summary/real_data.csv, paper_tmlr/tables/real_data.tex
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS as ALL_MODELS  # noqa: E402
from analyze_naturalistic import row  # noqa: E402

TASKS = [("dna_complement", "DNA", "Complement", "position-local"),
         ("dna_copy_previous", "DNA", "Copy previous base", "one-step"),
         ("dna_same_class", "DNA", "Same class as previous base", "one-step"),
         ("dna_gc_count", "DNA", "Running GC count mod 4", "prefix state"),
         ("text_shift", "Text", "Shift letters by one", "position-local"),
         ("text_copy_previous", "Text", "Copy previous letter", "one-step"),
         ("text_ascending", "Text", "Later than previous letter", "one-step"),
         ("text_vowel_count", "Text", "Running vowel count mod 4", "prefix state")]
# Only these three models were run on the real sequences
ALL_MODELS = [m for m in ALL_MODELS if m[0] in ("Qwen2.5 7B", "Qwen2.5 32B", "Gemma 4 31B")]
FORMATS = [("", "contiguous"), ("_spaced", "spaced")]


def main():
    rows = []
    for task, src, name, kind in TASKS:
        for label, tag in ALL_MODELS:
            if tag is None or str(tag).startswith("gemini"):
                continue
            for suf, fmt in FORMATS:
                d = load(task + suf, tag)
                if d is None:
                    continue
                r = row(task, label, d); r.update(source=src, name=name, kind=kind, format=fmt)
                rows.append(r)
    out = pd.DataFrame(rows).round(4)
    if out.empty:
        print("no results yet"); return
    out.to_csv(REPO / "results_clean" / "_summary" / "real_data.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[["task", "model", "format", "n", "base", "delta", "post_gain", "within_gain"]].to_string(index=False))

    lines = ["\\begin{tabular}{lll cc cc cc}", "\\toprule",
             "& & & \\multicolumn{2}{c}{No-marker acc.} & \\multicolumn{2}{c}{Pooled $\\Delta$} & \\multicolumn{2}{c}{Post-marker / within} \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
             "Source, task & Dependency & Model & Letters & Spaced & Letters & Spaced & Letters & Spaced \\\\", "\\midrule"]
    for task, src, name, kind in TASKS:
        first = True
        for label, _ in ALL_MODELS:
            x = out[(out.task == task) & (out.model == label)].set_index("format")
            if "contiguous" not in x.index or "spaced" not in x.index:
                continue
            c, s = x.loc["contiguous"], x.loc["spaced"]
            lines.append((f"{src}: {name}" if first else "") + f" & {kind if first else ''} & {label} & {c.base:.2f} & {s.base:.2f} & ${c.delta:+.2f}$ & ${s.delta:+.2f}$ & "
                         f"${c.post_gain:+.2f}$ / ${c.within_gain:+.2f}$ & ${s.post_gain:+.2f}$ / ${s.within_gain:+.2f}$ \\\\")
            first = False
        if not first:
            lines.append("\\addlinespace")
    while lines[-1] == "\\addlinespace":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "real_data.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/real_data.tex")


if __name__ == "__main__":
    main()
