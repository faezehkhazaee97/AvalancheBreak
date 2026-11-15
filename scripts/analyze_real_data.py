#!/usr/bin/env python3
"""Real sequences (tasks/dna_* and tasks/text_*): the three effects on windows of human promoter DNA
and of WikiText-2 letters, with a position-local, a one-step-dependency and a prefix-state task per
source, each in the contiguous and the spaced-character format. Per task, format and model:
no-marker accuracy, the pooled length-matched marker effect (k = 3 excluded, as in the main grid),
and the per-position comparison (post-marker and within-segment positions against the no-marker
run at the same indices; k in {2,4,8,16}; index scoring). Also the spacing-alone effect: the spaced
no-marker run against the contiguous no-marker run.

On the copy-previous tasks, also the share of outputs that are the start symbol (N for DNA, # for
text, which the rule gives only at the first position) at post-marker positions, at within-segment
positions, and at the same positions of the no-marker run: a model that restarts the rule at every
marker writes the start symbol after it.

Also an alignment-robust pass (--aligned runs it on its own): the same per-position comparison
re-scored with the edit-distance scorer of scripts/analyze_aligned_scoring.py, which is what the
synthetic tasks get in paper_tmlr/tables/aligned.tex. If the within-segment loss were cascading
misalignment after an off-by-one slip at the marker, aligning prediction to truth by minimum edit
distance would remove most of it, as it does on the synthetic dependency tasks.

Outputs: results_clean/_summary/real_data.csv, results_clean/_summary/real_data_restart.csv,
results_clean/_summary/real_data_aligned.csv,
paper_tmlr/tables/real_data.tex (the position-local and copy-previous tasks, main text),
paper_tmlr/tables/real_data_full.tex (all eight tasks, appendix),
paper_tmlr/tables/real_data_aligned.tex (index vs aligned scoring, the main-text four tasks)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS as ALL_MODELS  # noqa: E402
from analyze_naturalistic import row  # noqa: E402
from analyze_aligned_scoring import summarize, index_vector, aligned_vector  # noqa: E402

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

    write_table(out, [t for t in TASKS if t[3] == "position-local" or "copy_previous" in t[0]], "real_data.tex")
    write_table(out, TASKS, "real_data_full.tex")
    restart_shares()
    aligned_pass()


def write_table(out, tasks, fname):
    lines = ["\\begin{tabular}{lll cc cc cc}", "\\toprule",
             "& & & \\multicolumn{2}{c}{No-marker acc.} & \\multicolumn{2}{c}{Pooled $\\Delta$} & \\multicolumn{2}{c}{Post-marker / within} \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
             "Source, task & Dependency & Model & Contig. & Spaced & Contig. & Spaced & Contig. & Spaced \\\\", "\\midrule"]
    for task, src, name, kind in tasks:
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
    (REPO / "paper_tmlr" / "tables" / fname).write_text("\n".join(lines) + "\n")
    print("wrote tables/" + fname)


GAINS = ("base", "post", "within", "post_gain", "within_gain")


def share(idx, ali):
    """Share of an index-scored per-position effect that edit-distance alignment removes,
    (index - aligned) / index, as in scripts/make_tables_per_position.py. Undefined (NaN) where the
    index-scored effect is smaller than 0.02 in absolute value, or where the aligned effect has the
    opposite sign: the ratio then leaves [0, 1] and carries no meaning."""
    if abs(round(idx, 2)) < 0.02 or idx * ali < 0:
        return np.nan
    return (idx - ali) / idx


def aligned_pass():
    """Index vs edit-distance-aligned scoring of the real-sequence runs, cell by cell.

    Exactly the synthetic procedure of scripts/analyze_aligned_scoring.py: the same summarize() over
    N <= 64 and k in {2,4,8,16}, once with index_vector (what tables/real_data.tex reports) and once
    with aligned_vector (minimum-edit-distance alignment of prediction to truth), plus the share of
    each index-scored effect that alignment removes."""
    rows = []
    for task, src, name, kind in TASKS:
        for label, tag in ALL_MODELS:
            if tag is None or str(tag).startswith("gemini"):
                continue
            for suf, fmt in FORMATS:
                d = load(task + suf, tag)
                if d is None:
                    continue
                r = dict(source=src, task=task, name=name, kind=kind, model=label, format=fmt, n=len(d))
                for scorer, vec in (("index", index_vector), ("aligned", aligned_vector)):
                    s = summarize(d, vec)
                    r.update({f"{scorer}_{g}": s[g] for g in GAINS})
                r["post_share"] = share(r["index_post_gain"], r["aligned_post_gain"])
                r["within_share"] = share(r["index_within_gain"], r["aligned_within_gain"])
                rows.append(r)
                print(f"done {task}{suf} {label}", flush=True)
    if not rows:
        print("no results yet"); return
    out = pd.DataFrame(rows).round(4)
    out.to_csv(REPO / "results_clean" / "_summary" / "real_data_aligned.csv", index=False)
    pd.set_option("display.width", 240)
    print(out[["task", "model", "format", "index_post_gain", "aligned_post_gain",
               "index_within_gain", "aligned_within_gain", "within_share"]].to_string(index=False))
    write_aligned_table(out)
    cp = out[out.task.str.contains("copy_previous")]
    print("\ncopy_previous cells: median share of the index-scored within-segment effect that alignment removes: "
          f"{cp.within_share.median():.3f} ({cp.within_share.notna().sum()} of {len(cp)} cells defined)")


def signed(x):
    x = round(x, 2) + 0.0  # avoid "-0.00"
    return "$0.00$" if x == 0 else f"${'+' if x > 0 else '-'}{abs(x):.2f}$"


def write_aligned_table(out, fname="real_data_aligned.tex"):
    """One row per source/task/model/format: the index-scored and the aligned-scored post-marker and
    within-segment change, and the share of the within-segment change alignment removes. Same four
    tasks as tables/real_data.tex (position-local and copy-previous); the rest are in the CSV."""
    lines = [r"\begin{tabular}{lll cc ccc}", r"\toprule",
             r"& & & \multicolumn{2}{c}{Post-marker} & \multicolumn{3}{c}{Within segment} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"Source, task & Model & Format & Index & Aligned & Index & Aligned & Share \\", r"\midrule"]
    tasks = [t for t in TASKS if t[3] == "position-local" or "copy_previous" in t[0]]
    for task, src, name, kind in tasks:
        first = True
        for label, _ in ALL_MODELS:
            x = out[(out.task == task) & (out.model == label)].set_index("format")
            firstf = True
            for fmt, flab in (("contiguous", "Contig."), ("spaced", "Spaced")):
                if fmt not in x.index:
                    continue
                r = x.loc[fmt]
                s = r.within_share
                cell = "--" if pd.isna(s) else f"${'-' if s < 0 else ''}{abs(100 * s):.0f}\\%$"
                lines.append(f"{f'{src}: {name}' if first else ''} & {label if firstf else ''} & {flab} & "
                             f"{signed(r.index_post_gain)} & {signed(r.aligned_post_gain)} & "
                             f"{signed(r.index_within_gain)} & {signed(r.aligned_within_gain)} & {cell} \\\\")
                first = firstf = False
        lines.append(r"\addlinespace")
    while lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / fname).write_text("\n".join(lines) + "\n")
    print("wrote tables/" + fname)


START = {"dna_copy_previous": "N", "text_copy_previous": "#"}


def restart_shares():
    """Share of outputs equal to the start symbol at post-marker positions (i = k, 2k, ...), at
    within-segment positions, and at those same post-marker indices in the no-marker run; k >= 2,
    bar-stripped outputs indexed against the truth (index scoring, as in the per-position analysis)."""
    rows = []
    for task, sym in START.items():
        for label, tag in ALL_MODELS:
            for suf, fmt in FORMATS:
                f = REPO / "results_clean" / (task + suf) / "grid" / f"{tag}.csv"
                if not f.exists():
                    continue
                d = pd.read_csv(f, dtype={"LLM_Output": str, "Ground_Truth": str}, keep_default_na=False)
                cnt = {"post": [0, 0], "within": [0, 0], "nomarker_post": [0, 0]}
                for k in (2, 4, 8, 16):
                    for _, r in d[d.Interval.isin([0, k])].iterrows():
                        N, out = int(r.Seq_Length), str(r.LLM_Output)
                        if k >= N:
                            continue
                        for i in range(1, N):
                            if i % k == 0:
                                kind = "post" if r.Interval == k else "nomarker_post"
                            elif r.Interval == k:
                                kind = "within"
                            else:
                                continue
                            cnt[kind][1] += 1
                            cnt[kind][0] += int(i < len(out) and out[i] == sym)
                rows.append(dict(task=task, model=label, format=fmt,
                                 **{f"{a}_start": round(b[0] / b[1], 4) if b[1] else None for a, b in cnt.items()},
                                 n_post=cnt["post"][1]))
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(REPO / "results_clean" / "_summary" / "real_data_restart.csv", index=False)
        print("start symbol written (share of positions):"); print(t.to_string(index=False))


if __name__ == "__main__":
    # --aligned: only the alignment-robust pass (leaves real_data.tex and real_data_full.tex alone)
    aligned_pass() if "--aligned" in sys.argv else main()
