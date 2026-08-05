#!/usr/bin/env python3
"""Dose-response model-selection robustness, per reviewer Q1: "did you
attempt cross-validated model selection or information-criterion
comparisons that account for model complexity? Can you report confidence
intervals for the vertex location and its movement with N?"

For every individual boundary-density curve in the study (one curve per
task x model x length x axis_val cell, same k=3-excluded convention used
everywhere else), fits linear / quadratic / saturating-exponential, then:
  1. AIC and BIC (proper complexity penalty, not just R^2)
  2. Leave-one-density-out cross-validation (refit excluding each density
     point, predict it, sum squared error) -- directly tests whether
     quadratic's advantage is real signal or just more free parameters
     fitting noise on the SAME points used to fit it
  3. For quadratic fits: the vertex (peak) location, with a 500-iteration
     bootstrap CI
Also fits vertex-location ~ f(N) per task/model (the "practitioner recipe"
regression the reviewer's Q6 remainder is asking for -- what k to pick as
a function of sequence length).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tasks.common.compute_diagnostics import select_canonical_files  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results_clean"

MAIN_TASKS = [
    "cyclic_shift", "character_substitution", "random_permutation",
    "conditional_shift", "composed_permutation", "copy_previous", "running_max",
    "running_xor", "running_sum_local",
]


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def sat_exp(x, tar_max, b):
    return tar_max * (1 - np.exp(-b * x))


def fit_and_score(x, y):
    """Returns dict of per-family AIC/BIC/params, or None if too few points."""
    n = len(x)
    if n < 5:
        return None
    results = {}
    for name, fn, p0, nparams in [
        ("linear", linear, [0.0, 0.5], 2),
        ("quadratic", quadratic, [0.0, 0.0, 0.5], 3),
        ("exponential", sat_exp, [0.8, 1.0], 2),
    ]:
        try:
            popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=5000)
            pred = fn(x, *popt)
            rss = np.sum((y - pred) ** 2)
            rss = max(rss, 1e-12)
            aic = n * np.log(rss / n) + 2 * nparams
            bic = n * np.log(rss / n) + nparams * np.log(n)
            results[name] = {"params": popt, "rss": rss, "aic": aic, "bic": bic, "nparams": nparams}
        except Exception:
            results[name] = None
    return results


def loo_density_cv(x, y, fn, p0):
    """Leave-one-density-out CV: for each point, refit excluding it,
    predict it, accumulate squared error. Returns mean CV squared error,
    or None if too few points remain after leave-one-out."""
    n = len(x)
    if n < 6:
        return None
    errs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            popt, _ = curve_fit(fn, x[mask], y[mask], p0=p0, maxfev=5000)
            pred_i = fn(np.array([x[i]]), *popt)[0]
            errs.append((pred_i - y[i]) ** 2)
        except Exception:
            continue
    return np.mean(errs) if errs else None


def vertex_of_quadratic(a, b, c):
    if a == 0:
        return None
    return -b / (2 * a)


def bootstrap_vertex_ci(x, y, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vertices = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bx, by = x[idx], y[idx]
        if len(np.unique(bx)) < 3:
            continue
        try:
            popt, _ = curve_fit(quadratic, bx, by, p0=[0.0, 0.0, 0.5], maxfev=5000)
            v = vertex_of_quadratic(*popt)
            if v is not None and -0.5 <= v <= 1.5:  # sane range for 1/k in [0,1]
                vertices.append(v)
        except Exception:
            continue
    if len(vertices) < 20:
        return None, None
    return np.percentile(vertices, 2.5), np.percentile(vertices, 97.5)


def build_curve(df, length, axis_val, axis_col, exclude_k3=True):
    sub = df[df.Seq_Length == length]
    if axis_col:
        sub = sub[sub[axis_col] == axis_val]
    if exclude_k3:
        sub = sub[sub.Interval != 3]
    sub = sub.copy()
    sub["Inv_K"] = sub["Interval"].apply(lambda k: 1 / k if k > 0 else 0)
    agg = sub.groupby("Inv_K")["TAR"].mean().reset_index().sort_values("Inv_K")
    return agg["Inv_K"].to_numpy(), agg["TAR"].to_numpy()


def main():
    out_dir = RESULTS / "_dose_response_robustness"
    out_dir.mkdir(exist_ok=True)

    all_rows = []
    for task in MAIN_TASKS:
        model_files = select_canonical_files(task)
        for model, paths in model_files.items():
            frames = [pd.read_csv(p) for p in paths]
            df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            key_cols = [c for c in ["Method", "Interval", "Seq_Length", "Alphabet_Size", "Modulus", "Trial"] if c in df.columns]
            df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)

            axis_col = "Alphabet_Size" if "Alphabet_Size" in df.columns else ("Modulus" if "Modulus" in df.columns else None)
            axis_values = sorted(df[axis_col].unique()) if axis_col else [None]
            lengths = sorted(df.Seq_Length.unique())

            for length in lengths:
                for axis_val in axis_values:
                    x, y = build_curve(df, length, axis_val, axis_col)
                    if len(x) < 5:
                        continue
                    scores = fit_and_score(x, y)
                    if scores is None:
                        continue

                    row = {"Task": task, "Model": model, "Seq_Length": length,
                           "Axis_Val": axis_val, "N_Densities": len(x)}
                    for name in ["linear", "quadratic", "exponential"]:
                        r = scores.get(name)
                        if r is not None:
                            row[f"{name}_AIC"] = r["aic"]
                            row[f"{name}_BIC"] = r["bic"]

                    # AIC/BIC winner
                    valid = {k: v["aic"] for k, v in scores.items() if v is not None}
                    row["AIC_Winner"] = min(valid, key=valid.get) if valid else None
                    valid_bic = {k: v["bic"] for k, v in scores.items() if v is not None}
                    row["BIC_Winner"] = min(valid_bic, key=valid_bic.get) if valid_bic else None

                    # Leave-one-density-out CV: quadratic vs linear
                    cv_quad = loo_density_cv(x, y, quadratic, [0.0, 0.0, 0.5])
                    cv_lin = loo_density_cv(x, y, linear, [0.0, 0.5])
                    row["CV_MSE_Quadratic"] = cv_quad
                    row["CV_MSE_Linear"] = cv_lin
                    row["CV_Favors_Quadratic"] = (cv_quad is not None and cv_lin is not None and cv_quad < cv_lin)

                    # Vertex + bootstrap CI (quadratic fit only)
                    if scores.get("quadratic") is not None:
                        a, b, c = scores["quadratic"]["params"]
                        vertex = vertex_of_quadratic(a, b, c)
                        row["Vertex"] = vertex
                        if vertex is not None and -0.5 <= vertex <= 1.5:
                            ci_lo, ci_hi = bootstrap_vertex_ci(x, y)
                            row["Vertex_CI_Low"] = ci_lo
                            row["Vertex_CI_High"] = ci_hi

                    all_rows.append(row)

    out = pd.DataFrame(all_rows)
    out.to_csv(out_dir / "dose_response_robustness_full.csv", index=False)
    print(f"Wrote {len(out)} curves to {out_dir}/dose_response_robustness_full.csv")

    print("\n=== AIC winner distribution ===")
    print(out["AIC_Winner"].value_counts())
    print("\n=== BIC winner distribution ===")
    print(out["BIC_Winner"].value_counts())
    print("\n=== Leave-one-density-out CV: does quadratic beat linear on HELD-OUT points? ===")
    n_cv_valid = out["CV_Favors_Quadratic"].notna().sum()
    n_cv_favors_quad = (out["CV_Favors_Quadratic"] == True).sum()
    print(f"  Quadratic has lower held-out CV error in {n_cv_favors_quad}/{n_cv_valid} curves with valid CV")

    # Vertex-vs-N regression per task/model (practitioner recipe) -- UNGATED
    # version, kept for reference/comparison. This pools every curve with a
    # valid in-range vertex regardless of whether quadratic actually beats
    # linear on held-out data for that curve -- i.e. it includes curves where
    # the quadratic fit (and hence its vertex) may just be overfitting noise.
    print("\n=== Vertex location vs N, UNGATED (every valid-vertex curve) ===")
    vertex_df = out.dropna(subset=["Vertex"])
    vertex_df = vertex_df[(vertex_df.Vertex >= 0) & (vertex_df.Vertex <= 1)]

    def fit_recipe(df):
        rows = []
        for (task, model), sub in df.groupby(["Task", "Model"]):
            if len(sub) < 4 or sub.Seq_Length.nunique() < 3:
                continue
            slope, intercept = np.polyfit(sub.Seq_Length, sub.Vertex, 1)
            pred_vertex_at = {N: slope * N + intercept for N in [8, 16, 32, 64]}
            rows.append({
                "Task": task, "Model": model, "N_points": len(sub),
                "Slope": slope, "Intercept": intercept,
                **{f"PredictedVertex_N{N}": v for N, v in pred_vertex_at.items()},
            })
        return pd.DataFrame(rows)

    recipe_df = fit_recipe(vertex_df)
    recipe_df.to_csv(out_dir / "vertex_vs_N_recipe.csv", index=False)
    print(recipe_df.round(4).to_string(index=False))
    print(f"\nWrote {out_dir}/vertex_vs_N_recipe.csv ({len(recipe_df)} task/model recipes)")

    # CV-GATED version: only curves where leave-one-density-out CV actually
    # prefers quadratic over linear on held-out points (CV_Favors_Quadratic
    # == True) contribute a vertex to the regression -- the honest version of
    # this recipe, since an untested quadratic vertex isn't a real "optimal
    # k," just wherever the unregularized fit happened to curve.
    print("\n=== Vertex location vs N, CV-GATED (CV_Favors_Quadratic == True only) ===")
    gated_df = vertex_df[vertex_df.CV_Favors_Quadratic == True]  # noqa: E712
    print(f"  {len(gated_df)}/{len(vertex_df)} valid-vertex curves survive the CV gate")
    gated_recipe_df = fit_recipe(gated_df)
    gated_recipe_df.to_csv(out_dir / "vertex_vs_N_recipe_cv_gated.csv", index=False)
    if len(gated_recipe_df):
        print(gated_recipe_df.round(4).to_string(index=False))
    else:
        print("  No (Task, Model) group retains >=4 points / >=3 distinct N after gating --"
              " too sparse to fit a per-task/model recipe this way.")
    print(f"\nWrote {out_dir}/vertex_vs_N_recipe_cv_gated.csv ({len(gated_recipe_df)} task/model recipes)")


if __name__ == "__main__":
    main()
