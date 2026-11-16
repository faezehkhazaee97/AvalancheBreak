#!/usr/bin/env python3
"""Bootstrap intervals for the activation-patching recovery ratios.

A reviewer noted that Tables 6 and 31 report recovery ratios with no uncertainty attached, and
that some of the single-layer margins they are read against are small (9% against 6% on a 4.2-nat
denominator over 20 inputs). The ratio

    R = (patched - recipient) / (donor - recipient)

is computed from three means over the same 20 teacher-forced inputs, so the inputs are the
resampling unit: we draw 20 inputs with replacement 10,000 times, recompute all three means and
hence R within each draw, and report the 2.5th and 97.5th percentiles. Because the denominator is
resampled too, the interval widens when the clean--marked gap is itself uncertain, which is the
honest behaviour for a ratio.

Covers both experiments:
  results/activation_patching/<tag>.csv  (donor = the no-marker run; sites vs "clean"/"marked")
  results/patching_aligned/<tag>.csv     (donor = the across-wording marked run)

Output: results_clean/_summary/patching_intervals.csv
Usage:  python scripts/patching_bootstrap.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

B = 10000
SEED = 20261116
# (directory, label for the baseline row, label for the reference row)
SOURCES = [
    ("activation_patching", "marked", "clean"),
    ("patching_aligned", "recipient", "donor"),
]


def boot_ratio(patched, base, ref, rng):
    """Percentile interval for (mean(patched) - mean(base)) / (mean(ref) - mean(base)),
    resampling the shared input index."""
    n = len(patched)
    idx = rng.integers(0, n, size=(B, n))
    p, b, r = patched[idx].mean(1), base[idx].mean(1), ref[idx].mean(1)
    gap = r - b
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = np.where(np.abs(gap) > 1e-9, (p - b) / gap, np.nan)
    rat = rat[np.isfinite(rat)]
    if len(rat) < B // 10:
        return float("nan"), float("nan")
    return float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5))


def main():
    rng = np.random.default_rng(SEED)
    out = []
    for sub, base_name, ref_name in SOURCES:
        for f in sorted((REPO / "results" / sub).glob("*.csv")):
            d = pd.read_csv(f)
            model = f.stem
            for task, x in d.groupby("Task"):
                # align the three series on the trials present in all of them
                piv = x.pivot_table(index="Trial", columns=["Site", "Layer"], values="logodds")
                base_col = [c for c in piv.columns if c[0] == base_name]
                ref_col = [c for c in piv.columns if c[0] == ref_name]
                if not base_col or not ref_col:
                    continue
                for col in piv.columns:
                    site, layer = col
                    if site in (base_name, ref_name):
                        continue
                    sel = piv[[col, base_col[0], ref_col[0]]].dropna()
                    if len(sel) < 5:
                        continue
                    p, b, r = (sel[col].to_numpy(), sel[base_col[0]].to_numpy(),
                               sel[ref_col[0]].to_numpy())
                    point = (p.mean() - b.mean()) / (r.mean() - b.mean()) if abs(r.mean() - b.mean()) > 1e-9 else np.nan
                    lo, hi = boot_ratio(p, b, r, rng)
                    out.append(dict(experiment=sub, model=model, task=task, site=site,
                                    layer=layer, n_trials=len(sel), recovery=point,
                                    lo=lo, hi=hi))
    if not out:
        print("no patching results found")
        return
    o = pd.DataFrame(out).round(4)
    dest = REPO / "results_clean" / "_summary" / "patching_intervals.csv"
    o.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(o)} rows, {B} resamples, seed {SEED})")

    pd.set_option("display.width", 220)
    for sub, _, _ in SOURCES:
        a = o[(o.experiment == sub) & (o.layer.astype(str) == "all")]
        if a.empty:
            continue
        print(f"\n=== {sub}: all-layer recovery with 95% intervals ===")
        for _, r in a.sort_values(["model", "task", "site"]).iterrows():
            print(f"  {r.model:24s} {r.task:18s} {r.site:9s} "
                  f"{100 * r.recovery:6.0f}%  [{100 * r.lo:5.0f}, {100 * r.hi:5.0f}]")


if __name__ == "__main__":
    main()
