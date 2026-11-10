"""Length-matched pooled marker effect.

Only spacings k < N are run, so marker runs are spread unevenly over lengths (e.g. 2 spacings
at N = 4, 5 at N = 64) while no-marker runs are spread evenly. Accuracy falls with length, so
comparing the raw means of the two conditions biases the effect against markers. Here the
no-marker mean is reweighted to the marker runs' length distribution:

    Base_matched = sum_N p_N * mean TAR(Base at N),   p_N = share of marker rows at length N
    Delta        = mean TAR(Bounds) - Base_matched
    SE(Delta)    = sqrt( SE(Bounds)^2 + sum_N p_N^2 * SE(Base at N)^2 )

Rows with k = 3 are excluded (the paper's spacing grid is powers of two).
"""

import numpy as np


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
    se = np.sqrt(se_bounds ** 2 + se_base ** 2)
    return dict(base=base_m, bounds=bounds_m, delta=delta, z=delta / se if se > 0 else np.nan,
                base_unweighted=float(b[tar_col].mean()), n_base=len(b), n_bounds=len(c))
