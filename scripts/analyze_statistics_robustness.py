#!/usr/bin/env python3
"""Two robustness checks on how the pooled effects are estimated (reviewer questions).

1. Resampling instead of analytic standard errors. The paper's pooled effect is

       Delta = mean TAR(Bounds) - sum_N p_N * mean TAR(Base at N),
       p_N   = share of marker generations at length N,

   with SE(Delta) = sqrt(SE(Bounds)^2 + sum_N p_N^2 SE(Base at N)^2), which treats generations
   as independent (tasks/common/pooled.py). Each generation does use its own input, drawn from
   its own seed, but the marker generations pool cells of different difficulty (lengths,
   spacings, alphabets), so this script recomputes every pooled cell two further ways:

     gen      bootstrap over generations, stratified by condition and length
     cluster  bootstrap over whole (length, spacing, alphabet) cells, which lets whole cells
              enter or leave a resample and so propagates between-cell heterogeneity

   For each it reports the 95% interval and a bootstrap z (Delta / bootstrap SD), next to the
   analytic z, and whether the Bonferroni verdict (|z| >= 3.51) would change.

2. Mathematically redundant configurations. At A = 2 the three position-local rules coincide
   (all flip the bit) and at M = 2 running_sum_local computes running_xor, so those cells are
   not independent evidence across tasks. Every pooled cell is recomputed with A = 2 and
   M = 2 dropped, and the script reports how far the headline numbers move.

Output: results_clean/_summary/statistics_robustness.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from tasks.common.pooled import pooled_delta  # noqa: E402
from analyze_per_position import load, MODELS  # noqa: E402

LENGTHS = [4, 8, 16, 32, 64]
FAMILIES = {
    "letters": ["cyclic_shift", "random_permutation", "conditional_shift"],
    "spaced": ["cyclic_shift_spaced", "random_permutation_spaced", "conditional_shift_spaced"],
    "digits": ["cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"],
    "dependency": ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"],
}
B = 2000
RNG = np.random.default_rng(20261111)


def axis_col(d):
    return next((c for c in ("Alphabet_Size", "Modulus") if c in d.columns), None)


def prep(d):
    d = d[d.Interval.astype(int) != 3].copy()
    d["N"] = d.Seq_Length.astype(int)
    d["k"] = d.Interval.astype(int)
    d["TAR"] = d.TAR.astype(float)
    a = axis_col(d)
    d["axis"] = d[a].astype(int) if a else 0
    return d[d.N.isin(LENGTHS)]


def _cells(d):
    """Per (length, spacing, alphabet) cell: its length, the sum of its accuracies, its size."""
    g = d.groupby(["N", "k", "axis"]).TAR.agg(["sum", "count"]).reset_index()
    return g.N.to_numpy(), g["sum"].to_numpy(float), g["count"].to_numpy(float)


def bootstrap(d, mode, lengths):
    """Vectorized: returns B resampled values of the length-matched estimator."""
    base, bnd = d[d.Method == "Base"], d[d.Method == "Cellular_Bounds"]
    li = {N: i for i, N in enumerate(lengths)}
    if mode == "gen":
        # resample generations within each (condition, length); per-length counts, so the
        # matching weights p_N, stay fixed
        pN, bm, cm = [], [], []
        for N in lengths:
            b = base[base.N == N].TAR.to_numpy()
            c = bnd[bnd.N == N].TAR.to_numpy()
            if not len(c):
                continue
            if not len(b):
                return np.array([])
            pN.append(len(c))
            bm.append(b[RNG.integers(0, len(b), (B, len(b)))].mean(axis=1))
            cm.append(c[RNG.integers(0, len(c), (B, len(c)))].mean(axis=1))
        w = np.array(pN, float) / sum(pN)
        return (np.array(cm).T * w).sum(axis=1) - (np.array(bm).T * w).sum(axis=1)
    # resample whole cells within each condition; which lengths appear, and p_N, then vary
    bN, bS, bC = _cells(base)
    cN, cS, cC = _cells(bnd)
    bL, cL = np.array([li[n] for n in bN]), np.array([li[n] for n in cN])
    nL = len(lengths)
    out = np.full(B, np.nan)
    for i in range(B):
        bi, ci = RNG.integers(0, len(bL), len(bL)), RNG.integers(0, len(cL), len(cL))
        bs = np.bincount(bL[bi], weights=bS[bi], minlength=nL)
        bc = np.bincount(bL[bi], weights=bC[bi], minlength=nL)
        cs = np.bincount(cL[ci], weights=cS[ci], minlength=nL)
        cc = np.bincount(cL[ci], weights=cC[ci], minlength=nL)
        use = cc > 0
        if (bc[use] == 0).any():
            continue  # a length the marker runs need is absent from the no-marker resample
        p = cc[use] / cc[use].sum()
        out[i] = (cs[use] / cc[use] * p).sum() - (bs[use] / bc[use] * p).sum()
    return out[~np.isnan(out)]


def main():
    rows = []
    for fam, tasks in FAMILIES.items():
        for task in tasks:
            for label, tag in MODELS:
                if tag is None or str(tag).startswith("gemini"):
                    continue
                raw = load(task, tag)
                if raw is None:
                    continue
                d = prep(raw)
                r = pooled_delta(d, lengths=LENGTHS)
                if r is None:
                    continue
                rec = dict(family=fam, task=task, model=label, delta=r["delta"], z=r["z"])
                for mode in ("gen", "cluster"):
                    bs = bootstrap(d, mode, sorted(d.N.unique()))
                    if len(bs) < 100:
                        continue
                    sd = bs.std(ddof=1)
                    rec[f"{mode}_lo"], rec[f"{mode}_hi"] = np.percentile(bs, [2.5, 97.5])
                    rec[f"{mode}_z"] = r["delta"] / sd if sd > 0 else np.nan
                # the same cell with the redundant configurations dropped
                drop = d[~(((task in FAMILIES["letters"] + FAMILIES["spaced"] + FAMILIES["digits"])
                            & (d.axis == 2)) | ((task == "running_sum_local") & (d.axis == 2)))]
                r2 = pooled_delta(drop, lengths=LENGTHS) if len(drop) else None
                rec["delta_no_redundant"] = r2["delta"] if r2 else np.nan
                rec["z_no_redundant"] = r2["z"] if r2 else np.nan
                rows.append(rec)
        print("done", fam, flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(REPO / "results_clean" / "_summary" / "statistics_robustness.csv", index=False)

    sig = out.z.abs() >= 3.51
    print(f"\ncells: {len(out)}; significant analytically: {sig.sum()}")
    for mode in ("gen", "cluster"):
        s2 = out[f"{mode}_z"].abs() >= 3.51
        print(f"  {mode:8s} bootstrap: significant {s2.sum()};  verdict differs in {(sig != s2).sum()} cells;"
              f"  median |z| ratio {np.nanmedian(out[f'{mode}_z'].abs() / out.z.abs()):.2f}")
    excl = out.dropna(subset=["delta_no_redundant"])
    diff = (excl.delta_no_redundant - excl.delta).abs()
    s3 = excl.z_no_redundant.abs() >= 3.51
    print(f"\ndropping A = 2 and M = 2 ({len(excl)} cells): median |change| in delta {diff.median():.3f}, "
          f"max {diff.max():.3f}; sign flips {(np.sign(excl.delta_no_redundant) != np.sign(excl.delta)).sum()}; "
          f"significance differs in {(s3 != (excl.z.abs() >= 3.51)).sum()} cells")
    print("\nlargest shifts when the redundant configurations are dropped:")
    print(excl.assign(change=diff).nlargest(6, "change")[
        ["family", "task", "model", "delta", "delta_no_redundant", "z", "z_no_redundant"]].to_string(index=False))


if __name__ == "__main__":
    main()
