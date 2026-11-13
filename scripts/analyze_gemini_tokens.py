#!/usr/bin/env python3
"""What Gemini's own token accounting says about how it tokenizes our inputs.

We cannot inspect Gemini's tokenizer, but every API response reports the prompt's token count.
Across the Gemini 3.6 grid the instruction is fixed while the input grows, so differences of
prompt token counts between lengths give the tokens added per added input character, and
differences between spacings at one length give the tokens added per marker.

Reads results_clean/gemini_pilot/gemini-3.6-flash_POOLED.csv, prints the per-step values, and
writes paper_tmlr/tables/gemini_tokens.tex (tokens per added character at the 32 -> 64 step,
net of one token per marker, by input format and spacing)."""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results_clean" / "gemini_pilot" / "gemini-3.6-flash_POOLED.csv"
OUT = REPO / "paper_tmlr" / "tables" / "gemini_tokens.tex"

FORMATS = [
    ("Letters", ["cyclic_shift", "random_permutation", "conditional_shift"]),
    ("Spaced letters", ["cyclic_shift_spaced", "random_permutation_spaced", "conditional_shift_spaced"]),
    ("Digits", ["cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"]),
    ("Binary digits (dependency tasks)", ["running_xor", "running_sum_local", "copy_previous", "sliding_window_max"]),
]
SPACINGS = [0, 1, 2, 4, 8]
STEPS = [(4, 8), (8, 16), (16, 32), (32, 64)]


def main():
    d = pd.read_csv(SRC)
    assert (d.Model_Resolved == "gemini-3.6-flash").all()
    d = d.assign(N=d.Seq_Length.astype(int), k=d.Interval.astype(int),
                 M=d.Input_Given.astype(str).str.count(r"\|"))
    # Mean prompt tokens and marker count per (task, N, k). Within a task the item-to-item spread
    # comes from the instruction (e.g. the shift amount) and is the same at every N and k, so
    # differences of these means isolate the input's tokens.
    g = d.groupby(["Task", "N", "k"]).agg(tok=("Prompt_Tokens", "mean"), M=("M", "mean")).reset_index()

    rows, per_marker = [], {}
    for fmt, tasks in FORMATS:
        x = g[g.Task.isin(tasks)]
        # tokens per marker: (k, N) minus (0, N), divided by the marker count, over all N and k > 0
        pm = []
        for t in tasks:
            for n in sorted(x.N.unique()):
                base = x[(x.Task == t) & (x.N == n) & (x.k == 0)]
                for _, r in x[(x.Task == t) & (x.N == n) & (x.k > 0)].iterrows():
                    if not base.empty and r.M > 0:
                        pm.append((r.tok - base.tok.iloc[0]) / r.M)
        per_marker[fmt] = (min(pm), max(pm))
        cells = {}
        for k in SPACINGS:
            vals = {}
            for lo, hi in STEPS:
                a = x[(x.N == lo) & (x.k == k)].groupby("Task")[["tok", "M"]].mean()
                b = x[(x.N == hi) & (x.k == k)].groupby("Task")[["tok", "M"]].mean()
                both = a.index.intersection(b.index)
                if len(both) == 0:
                    continue
                dt = (b.loc[both, "tok"] - a.loc[both, "tok"]).mean()
                dm = (b.loc[both, "M"] - a.loc[both, "M"]).mean()
                vals[(lo, hi)] = (dt - dm) / (hi - lo)   # net of one token per marker
            cells[k] = vals
        rows.append((fmt, cells))
        print(fmt)
        print("  tokens per marker: min %.3f max %.3f" % per_marker[fmt])
        for k in SPACINGS:
            print("  k=%d: " % k + "  ".join(f"{lo}->{hi}: {v:.3f}" for (lo, hi), v in cells[k].items()))

    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Input format & $k = 0$ & $k = 1$ & $k = 2$ & $k = 4$ & $k = 8$ \\", r"\midrule"]
    for fmt, cells in rows:
        vals = [cells[k].get((32, 64), np.nan) for k in SPACINGS]
        lines.append(fmt + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
