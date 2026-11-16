#!/usr/bin/env python3
"""Real sequences (tasks/dna_* and tasks/text_*): the three effects on windows of human promoter DNA
and of WikiText-2 letters, with a position-local, a one-step-dependency and a prefix-state task per
source, each in the contiguous and the spaced-character format. Per task, format and model:
no-marker accuracy, the pooled length-matched marker effect (k = 3 excluded, as in the main grid),
and the per-position comparison (post-marker and within-segment positions against the no-marker
run at the same indices; k in {2,4,8,16}), scored two ways in the same row:
  index    index-by-index, the paper's TAR;
  aligned  the edit-distance scorer of scripts/analyze_aligned_scoring.py, which is what the
           synthetic tasks get in paper_tmlr/tables/aligned.tex.
If the within-segment loss were cascading misalignment after an off-by-one slip at the marker,
aligning prediction to truth by minimum edit distance would remove most of it, as it does on the
synthetic dependency tasks. "Share" is the part of the index-scored within-segment effect that
alignment removes.

One run produces both tables: the four main-text rules (position-local and copy-previous) and,
without repeating them, the four remaining rules. Also the spacing-alone effect: the spaced
no-marker run against the contiguous no-marker run.

On the copy-previous tasks, also the share of outputs that are the start symbol (N for DNA, # for
text, which the rule gives only at the first position) at post-marker positions, at within-segment
positions, and at the same positions of the no-marker run: a model that restarts the rule at every
marker writes the start symbol after it.

Outputs: results_clean/_summary/real_data.csv, results_clean/_summary/real_data_restart.csv,
paper_tmlr/tables/real_data.tex (the position-local and copy-previous rules, main text),
paper_tmlr/tables/real_data_full.tex (the other four rules, appendix)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS as ALL_MODELS  # noqa: E402
from analyze_naturalistic import pooled_delta  # noqa: E402
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
FORMATS = [("", "contiguous", "Contig."), ("_spaced", "spaced", "Spaced")]
# Table 3 carries every rule Section 7 analyses: the position-local ones and all three one-step
# ones. Same-class and later-letter were in the appendix before, but the section reads their
# spaced rows, so the section was not self-contained without them. The appendix keeps the two
# prefix-state counting rules, which stay near chance in both formats.
MAIN = [t for t in TASKS if t[3] in ("position-local", "one-step")]
REST = [t for t in TASKS if t not in MAIN]
GAINS = ("base", "post", "within", "post_gain", "within_gain")


def main():
    rows = []
    for task, src, name, kind in TASKS:
        for label, tag in ALL_MODELS:
            if tag is None or str(tag).startswith("gemini"):
                continue
            for suf, fmt, _ in FORMATS:
                d = load(task + suf, tag)
                if d is None:
                    continue
                r = dict(task=task, source=src, name=name, kind=kind, model=label, format=fmt,
                         n=len(d), delta=pooled_delta(d))
                for scorer, vec in (("index", index_vector), ("aligned", aligned_vector)):
                    s = summarize(d, vec)
                    r.update({f"{scorer}_{g}": s[g] for g in GAINS})
                # the index-scored columns keep their plain names, as before the merge
                r.update(base=r["index_base"], post=r["index_post"],
                         post_gain=r["index_post_gain"], within_gain=r["index_within_gain"])
                r["post_share"] = share(r["index_post_gain"], r["aligned_post_gain"])
                r["within_share"] = share(r["index_within_gain"], r["aligned_within_gain"])
                rows.append(r)
                print(f"done {task}{suf} {label}", flush=True)
    if not rows:
        print("no results yet"); return
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPO / "results_clean" / "_summary" / "real_data.csv", index=False)
    pd.set_option("display.width", 240)
    print(out[["task", "model", "format", "n", "base", "delta", "index_post_gain", "aligned_post_gain",
               "index_within_gain", "aligned_within_gain", "within_share"]].round(4).to_string(index=False))

    # NB: the tables are formatted from the unrounded frame. Rounding to four decimals first made
    # a cell such as 0.235037 print as +0.23 here and +0.24 in the aligned table, because the
    # double nearest 0.235 lies just below the tie (see the module docstring of this change).
    write_table(out, MAIN, "real_data.tex")
    write_table(out, REST, "real_data_full.tex")
    restart_shares()
    cp = out[out.task.str.contains("copy_previous")]
    print("\ncopy_previous cells: median share of the index-scored within-segment effect that alignment removes: "
          f"{cp.within_share.median():.3f} ({cp.within_share.notna().sum()} of {len(cp)} cells defined)")


def signed(x):
    """Two decimals with an explicit sign, rounded from the unrounded value."""
    x = round(float(x), 2) + 0.0  # avoid "-0.00"
    return "$0.00$" if x == 0 else f"${'+' if x > 0 else '-'}{abs(x):.2f}$"


def share(idx, ali):
    """Share of an index-scored per-position effect that edit-distance alignment removes,
    (index - aligned) / index, as in scripts/make_tables_per_position.py. Undefined (NaN) where the
    index-scored effect is smaller than 0.02 in absolute value, or wherever the ratio leaves [0, 1]
    and so carries no meaning as a share: that happens when the aligned effect has the opposite sign
    and also when alignment makes the effect larger rather than smaller (which the near-chance rules
    of the full table do, giving shares like -237%)."""
    if abs(round(idx, 2)) < 0.02 or idx * ali < 0:
        return np.nan
    s = (idx - ali) / idx
    return s if 0.0 <= s <= 1.0 else np.nan


def write_table(out, tasks, fname):
    """One row per source/task/model/format. Columns:
       1 Source, task | 2 Dependency | 3 Model | 4 Format | 5 No-marker acc. | 6 Pooled Delta
       | 7 Post-marker, index | 8 Post-marker, aligned
       | 9 Within segment, index | 10 Within segment, aligned | 11 Share
    Bold (scripts/bold_tables.py): column 7, index-scored post-marker changes of 0.25 or more."""
    lines = [r"\begin{tabular}{llll cc cc cc c}", r"\toprule",
             r"& & & & & & \multicolumn{2}{c}{Post-marker} & \multicolumn{3}{c}{Within segment} \\",
             r"\cmidrule(lr){7-8}\cmidrule(lr){9-11}",
             r"Source, task & Dependency & Model & Format & No-marker acc. & Pooled $\Delta$ & "
             r"Index & Aligned & Index & Aligned & Share \\", r"\midrule"]
    for task, src, name, kind in tasks:
        first = True
        for label, _ in ALL_MODELS:
            x = out[(out.task == task) & (out.model == label)].set_index("format")
            firstf = True
            for _, fmt, flab in FORMATS:
                if fmt not in x.index:
                    continue
                r = x.loc[fmt]
                s = r.within_share
                cell = "--" if pd.isna(s) else f"${'-' if s < 0 else ''}{abs(100 * s):.0f}\\%$"
                lines.append(f"{f'{src}: {name}' if first else ''} & {kind if first else ''} & "
                             f"{label if firstf else ''} & {flab} & {r.base:.2f} & {signed(r.delta)} & "
                             f"{signed(r.index_post_gain)} & {signed(r.aligned_post_gain)} & "
                             f"{signed(r.index_within_gain)} & {signed(r.aligned_within_gain)} & {cell} \\\\")
                first = firstf = False
        if not first:
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
            for suf, fmt, _ in FORMATS:
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
    main()
