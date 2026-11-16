#!/usr/bin/env python3
"""Does the spacing want to be fixed, or to scale with the sequence?

A reviewer asked whether an adaptive k -- sparser markers as the sequence grows -- would keep
the within-segment gain while reducing the blind spot, since a longer sequence then has no more
boundaries than a short one. The existing k-sweep already answers this: it holds every (N, k)
cell, so "k fixed at 4" and "k = N/8, eight segments whatever the length" are both paths through
the same grid and can be read off without new runs.

For each model and dependency task we follow the two policies across N and compare the
post-marker change and the within-segment change at the cell each policy selects.

Output: results_clean/_summary/adaptive_k.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from analyze_aligned_scoring import summarize, index_vector  # noqa: E402
from analyze_per_position import load, MODELS, horizon  # noqa: E402

TASKS = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]
LENGTHS = [32, 64]
SEGMENTS = 8


def main():
    rows = []
    for label, tag in MODELS:
        if tag is None or tag == "gemini36":
            continue
        for task in TASKS:
            d = load(task, tag)
            if d is None:
                continue
            d = d.copy()
            d["N"] = d.Seq_Length.astype(int)
            d["k"] = d.Interval.astype(int)
            for N in LENGTHS:
                for policy, k in (("fixed k=4", 4), ("adaptive k=N/8", max(2, N // SEGMENTS))):
                    cell = d[(d.N == N) & (d.k.isin([0, k]))]
                    if cell[cell.k == k].empty or cell[cell.k == 0].empty:
                        continue
                    s = summarize(cell, index_vector, horizon(task))
                    if not s or np.isnan(s.get("post_gain", np.nan)):
                        continue
                    rows.append(dict(model=label, task=task, N=N, policy=policy, k=k,
                                     post_gain=s["post_gain"], within_gain=s["within_gain"]))
    if not rows:
        print("no cells found")
        return
    o = pd.DataFrame(rows).round(4)
    out = REPO / "results_clean" / "_summary" / "adaptive_k.csv"
    o.to_csv(out, index=False)
    print(f"wrote {out} ({len(o)} rows)")
    pd.set_option("display.width", 200)
    piv = o.pivot_table(index=["N"], columns="policy", values=["post_gain", "within_gain"])
    print("\nmean over models and dependency tasks:")
    print(piv.round(3).to_string())
    print("\nat N = 64 the two policies differ (k = 4 against k = 8); at N = 32 they coincide.")
    n64 = o[o.N == 64]
    for pol, g in n64.groupby("policy"):
        print("  %-15s post %+.3f   within %+.3f" % (pol, g.post_gain.mean(), g.within_gain.mean()))


if __name__ == "__main__":
    main()
