"""Length-matched pooled marker effect.

Only spacings k < N are run, so marker runs are spread unevenly over lengths (e.g. 2 spacings
at N = 4, 5 at N = 64) while no-marker runs are spread evenly. Accuracy falls with length, so
comparing the raw means of the two conditions biases the effect against markers. Here the
no-marker mean is reweighted to the marker runs' length distribution:

    Base_matched = sum_N p_N * mean TAR(Base at N),   p_N = share of marker rows at length N
    Delta        = mean TAR(Bounds) - Base_matched

Standard error. The (N, k, A) cells of the grid are a fixed design, not a sample from a
population of cells, and accuracy varies steeply across them. Pooling all of a condition's
generations into one s^2/n therefore charges that deterministic between-cell variation to
sampling noise. The primary standard error treats each cell as a fixed stratum: with w_c the
share and n_c the generation count of marker generations in stratum c,

    SE(Bounds)^2    = sum_c w_c^2 s_c^2 / n_c      strata c = (N, k, A)
    SE(Base at N)^2 = sum_a w_a^2 s_a^2 / n_a      strata a = alphabet size, within N
    SE(Delta)       = sqrt( SE(Bounds)^2 + sum_N p_N^2 * SE(Base at N)^2 ),  z = Delta / SE

The unstratified form, s/sqrt(n) over all of a condition's generations, is still returned as
`se_unstrat` / `z_unstrat`; it is the more conservative of the two almost everywhere, and the
cell-level bootstrap (scripts/analyze_statistics_robustness.py) remains the conservative check.

Rows with k = 3 are excluded (the paper's spacing grid is powers of two).
"""

import numpy as np

AXIS_COLS = ("Alphabet_Size", "Modulus", "Axis_Val")


def axis_col(df):
    """The alphabet-size / modulus column of this task's grid, if it has one."""
    return next((c for c in AXIS_COLS if c in df.columns), None)


def stratified_var(d, by, tar_col="TAR"):
    """sum_c w_c^2 s_c^2 / n_c over the fixed strata `by`, with w_c = n_c / n.

    A stratum holding a single generation has no within-stratum variance of its own. It is
    given the pooled within-stratum variance rather than zero, so a thin stratum is never
    silently counted as noiseless. Returns (variance, number of such strata).
    """
    g = d.groupby(by, dropna=False, observed=True)[tar_col].agg(["var", "count"])
    if not len(g):
        return np.nan, 0
    w = g["count"] / g["count"].sum()
    big = g[g["count"] >= 2]
    dof = float((big["count"] - 1).sum())
    pooled = float(((big["count"] - 1) * big["var"]).sum() / dof) if dof > 0 else np.nan
    var = g["var"].where(g["count"] >= 2, pooled)
    thin = int((g["count"] < 2).sum())
    if var.isna().any():
        return np.nan, thin
    return float((w ** 2 * var / g["count"]).sum()), thin


def pooled_delta(df, lengths=None, method_col="Method", base="Base", bounds="Cellular_Bounds",
                 length_col="Seq_Length", interval_col="Interval", tar_col="TAR"):
    d = df[df[interval_col].astype(int) != 3]
    if lengths is not None:
        d = d[d[length_col].astype(int).isin(lengths)]
    b = d[d[method_col] == base]
    c = d[d[method_col] == bounds]
    if len(b) < 2 or len(c) < 2:
        return None
    p = c.groupby(length_col).size() / len(c)
    gb = b.groupby(length_col)[tar_col]
    mean_b, se_b = gb.mean().reindex(p.index), gb.sem().reindex(p.index)
    if mean_b.isna().any():
        return None
    base_m = float((p * mean_b).sum())
    se_base = float(np.sqrt((p ** 2 * se_b.fillna(0) ** 2).sum()))
    bounds_m, se_bounds = float(c[tar_col].mean()), float(c[tar_col].sem())
    delta = bounds_m - base_m
    se_unstrat = float(np.sqrt(se_bounds ** 2 + se_base ** 2))
    # primary: fixed (N, k, A) strata on the marker side, alphabets within each length on the
    # baseline side (the length weights p_N are held fixed, as in the unstratified form)
    a = axis_col(d)
    var_c, thin = stratified_var(c, [length_col, interval_col] + ([a] if a else []), tar_col)
    by_len = dict(tuple(b.groupby(length_col)))
    var_b = 0.0
    for N, pN in p.items():
        bn = by_len.get(N)
        v, t = (np.nan, 0) if bn is None else stratified_var(bn, [a] if a else [length_col], tar_col)
        thin += t
        if not np.isfinite(v):
            var_b = np.nan
            break
        var_b += float(pN) ** 2 * v
    se = float(np.sqrt(var_c + var_b)) if np.isfinite(var_c) and np.isfinite(var_b) else np.nan
    return dict(base=base_m, bounds=bounds_m, delta=delta, se=se,
                z=delta / se if np.isfinite(se) and se > 0 else np.nan,
                se_unstrat=se_unstrat,
                z_unstrat=delta / se_unstrat if se_unstrat > 0 else np.nan,
                thin_strata=thin,
                base_unweighted=float(b[tar_col].mean()), n_base=len(b), n_bounds=len(c))
