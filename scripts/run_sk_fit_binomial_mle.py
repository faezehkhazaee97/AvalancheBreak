"""Binomial-likelihood MLE alternative to the existing nonlinear-least-squares
SK (Sherrington-Kirkpatrick) fit, per reviewer request: "did you explore a
binomial-likelihood MLE ... that can handle near-zero SAR more robustly? If
so, do the alpha trends persist?"

SAR at each length N is itself a proportion of trials that were fully
correct (0/1 per trial) -- i.e. a binomial observation, not a continuous
measurement with additive Gaussian noise. The existing fit
(thermodynamics_sk_fit*.csv) minimizes sum-of-squares on the *aggregated*
SAR(N) points; this instead maximizes the binomial log-likelihood over the
*raw per-trial* 0/1 outcomes directly, which is the statistically correct
likelihood for this data and should be more robust exactly where SAR is
near 0 or 1 (few trials => high sampling noise on the aggregated proportion,
which least-squares treats as if it were a precise measurement).

Model (same as existing fit): SAR(N) = exp(-beta0 * N * alpha^(N-1))
Per-trial likelihood: Bernoulli(p=SAR(N)) for each trial at that N.

Usage: python scripts/run_sk_fit_binomial_mle.py <raw_grid_csv> <out_csv>
       e.g. python scripts/run_sk_fit_binomial_mle.py \
            results/Qwen2.5-7B-Instruct/thermodynamics_grid_templated_v2.csv \
            results/Qwen2.5-7B-Instruct/thermodynamics_sk_fit_binomial_mle.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import bootstrap

ALPHA_BOUNDS = (1 + 1e-6, 5.0)
BETA0_BOUNDS = (1e-8, 5.0)


def sar_model(N, beta0, alpha):
    return np.exp(-beta0 * N * np.power(alpha, N - 1))


def neg_log_likelihood(params, Ns, outcomes):
    beta0, alpha = params
    p = np.clip(sar_model(Ns, beta0, alpha), 1e-12, 1 - 1e-12)
    return -np.sum(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p))


def fit_binomial_mle(Ns, outcomes, n_boot=500, seed=0):
    """Ns, outcomes: parallel arrays, one row per trial (not aggregated)."""
    x0 = [0.05, 1.1]
    bounds = [BETA0_BOUNDS, ALPHA_BOUNDS]
    res = minimize(neg_log_likelihood, x0, args=(Ns, outcomes), bounds=bounds, method="L-BFGS-B")
    beta0_hat, alpha_hat = res.x
    pinned = (
        abs(alpha_hat - ALPHA_BOUNDS[0]) < 1e-4 or abs(alpha_hat - ALPHA_BOUNDS[1]) < 1e-4
    )

    rng = np.random.default_rng(seed)
    n = len(Ns)
    boot_alphas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bNs, bO = Ns[idx], outcomes[idx]
        try:
            r = minimize(neg_log_likelihood, x0, args=(bNs, bO), bounds=bounds, method="L-BFGS-B")
            boot_alphas.append(r.x[1])
        except Exception:
            continue
    boot_alphas = np.array(boot_alphas)
    ci_lo, ci_hi = (np.percentile(boot_alphas, 2.5), np.percentile(boot_alphas, 97.5)) if len(boot_alphas) > 10 else (np.nan, np.nan)

    return {
        "Beta0_MLE": beta0_hat, "Alpha_MLE": alpha_hat,
        "Alpha_CI_Low": ci_lo, "Alpha_CI_High": ci_hi,
        "Pinned_At_Bound": pinned, "N_trials": n, "N_boot_valid": len(boot_alphas),
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_sk_fit_binomial_mle.py <raw_grid_csv> <out_csv>")
        sys.exit(1)
    in_csv, out_csv = sys.argv[1], sys.argv[2]
    df = pd.read_csv(in_csv)

    group_cols = [c for c in ["Alphabet_Size", "Method", "Interval"] if c in df.columns]
    rows = []
    for key, sub in df.groupby(group_cols):
        Ns = sub["Seq_Length"].to_numpy(dtype=float)
        outcomes = sub["SAR"].to_numpy(dtype=float)
        # SK fit needs SAR non-degenerate (not all 0, not all 1) to be informative
        if outcomes.min() == outcomes.max():
            continue
        if len(np.unique(Ns)) < 3:
            continue
        fit = fit_binomial_mle(Ns, outcomes)
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(fit)
        rows.append(row)
        print(f"{row} ")

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"\nWrote {len(out)} rows to {out_csv}")


if __name__ == "__main__":
    main()
