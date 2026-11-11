#!/usr/bin/env python3
"""Interaction between input and output markers (reviewer question), plus marker-placement
compliance in the output-only condition.

For the 2x2 of scripts/run_marker_sides.py the interaction is

    Delta_int = A11 - A10 - A01 + A00

(both sides marked, minus input only, minus output only, plus neither). The three marked
conditions are built from one underlying string per (task, N, k, trial), so they are resampled
together as paired units; A00 comes from the main grid at k = 0, whose inputs are drawn from
different seeds, so it is resampled independently. 10,000 bootstrap replicates.

Also reports, for the output-only condition, how often the model actually produced markers at
the requested offsets.

Output: results_clean/_summary/marker_sides_interaction.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_marker_sides import grid, TASKS, LENGTHS, KS, TEXT  # noqa: E402
from analyze_compliance import marker_positions, expected_positions  # noqa: E402

B = 10000
RNG = np.random.default_rng(11112026)


def main():
    s = pd.read_csv(REPO / "results" / "marker_sides" / "Qwen2.5-32B-Instruct.csv", dtype=TEXT, keep_default_na=False)
    s = s.assign(N=s.Seq_Length.astype(int), k=s.Interval.astype(int), t=s.Trial.astype(int), TAR=s.TAR.astype(float))
    rows = []
    for task, ac, axis, lab in TASKS:
        a00 = grid(task, ac, axis, [0]).TAR.to_numpy()
        a11 = grid(task, ac, axis, KS).set_index(["N", "k", "t"]).TAR
        a10 = s[(s.Task == task) & (s.Condition == "in_only")].set_index(["N", "k", "t"]).TAR
        a01 = s[(s.Task == task) & (s.Condition == "out_only")].set_index(["N", "k", "t"]).TAR
        keys = a11.index.intersection(a10.index).intersection(a01.index)
        M = np.column_stack([a11.loc[keys].to_numpy(), a10.loc[keys].to_numpy(), a01.loc[keys].to_numpy()])
        point = M[:, 0].mean() - M[:, 1].mean() - M[:, 2].mean() + a00.mean()
        reps = np.empty(B)
        for i in range(B):
            r = M[RNG.integers(0, len(M), len(M))]
            b = a00[RNG.integers(0, len(a00), len(a00))]
            reps[i] = r[:, 0].mean() - r[:, 1].mean() - r[:, 2].mean() + b.mean()
        lo, hi = np.percentile(reps, [2.5, 97.5])
        # compliance in the output-only condition: markers requested but absent from the input
        oo = s[(s.Task == task) & (s.Condition == "out_only")]
        comp = np.mean([marker_positions(r.Raw_Model_Output) == expected_positions(len(r.Ground_Truth), r.k)
                        for r in oo.itertuples()])
        any_marker = np.mean([len(marker_positions(r.Raw_Model_Output)) > 0 for r in oo.itertuples()])
        rows.append(dict(task=task, n_paired=len(keys), interaction=point, lo=lo, hi=hi,
                         out_only_compliant=comp, out_only_any_marker=any_marker))
    o = pd.DataFrame(rows).round(4)
    o.to_csv(REPO / "results_clean" / "_summary" / "marker_sides_interaction.csv", index=False)
    print(o.to_string(index=False))


if __name__ == "__main__":
    main()
