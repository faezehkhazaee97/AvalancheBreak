#!/usr/bin/env python3
"""45-row aggregate dose-response table (Appendix E, top table): for every
(Task, Model) pair, fit linear/quadratic/saturating-exponential to EVERY
individual (Seq_Length, Axis_Val) curve, then report the MEAN R^2 across all
of that pair's curves -- the aggregate-level claim in Section 3.2 ("the
quadratic wins all 45 of 45 aggregate rows"), as distinct from the 348-row
per-curve table (compute_dose_response_robustness.py), which reports
AIC/BIC/CV per INDIVIDUAL curve rather than averaged up to (Task, Model).

Uses the same DOCUMENTED_MODELS-filtered curve set as the 348-row table
(reruns the identical fits fresh here to get R^2, which the 348-row script
doesn't compute) -- both tables are therefore built from the same
post-OLMo/72B-decontamination data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tasks.common.compute_diagnostics import select_canonical_files  # noqa: E402
from tasks.common.compute_dose_response_robustness import (  # noqa: E402
    MAIN_TASKS, linear, quadratic, sat_exp, build_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results_clean"
OUT_DIR = RESULTS / "_dose_response_robustness"


def fit_r2(x, y):
    n = len(x)
    if n < 5:
        return None
    ss_tot = np.sum((y - y.mean()) ** 2)
    out = {}
    for name, fn, p0 in [
        ("linear", linear, [0.0, 0.5]),
        ("quadratic", quadratic, [0.0, 0.0, 0.5]),
        ("exponential", sat_exp, [0.8, 1.0]),
    ]:
        try:
            popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=5000)
            pred = fn(x, *popt)
            ss_res = np.sum((y - pred) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            out[name] = r2
        except Exception:
            out[name] = float("nan")
    return out


def main():
    rows = []
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

            curve_r2 = {"linear": [], "quadratic": [], "exponential": []}
            n_curves = 0
            for length in lengths:
                for axis_val in axis_values:
                    x, y = build_curve(df, length, axis_val, axis_col)
                    if len(x) < 5:
                        continue
                    r2 = fit_r2(x, y)
                    if r2 is None:
                        continue
                    n_curves += 1
                    for k in curve_r2:
                        curve_r2[k].append(r2[k])

            if n_curves == 0:
                continue
            rows.append({
                "Task": task, "Model": model, "N_Curves": n_curves,
                "Mean_R2_Linear": np.nanmean(curve_r2["linear"]),
                "Mean_R2_Quadratic": np.nanmean(curve_r2["quadratic"]),
                "Mean_R2_Exponential": np.nanmean(curve_r2["exponential"]),
            })

    out = pd.DataFrame(rows)
    out["Quadratic_Wins"] = (out.Mean_R2_Quadratic > out.Mean_R2_Linear) & (out.Mean_R2_Quadratic > out.Mean_R2_Exponential)
    out_path = OUT_DIR / "dose_response_45row_aggregate.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")
    print(f"\nQuadratic wins {out.Quadratic_Wins.sum()}/{len(out)} aggregate rows")
    print(f"Mean R2 -- linear: {out.Mean_R2_Linear.mean():.3f}  quadratic: {out.Mean_R2_Quadratic.mean():.3f}  "
          f"exponential: {out.Mean_R2_Exponential.mean():.3f}")
    print(f"Exponential R2 range: {out.Mean_R2_Exponential.min():.1f} to {out.Mean_R2_Exponential.max():.3f}")
    print("\nPer-task mean quadratic R2 (across models):")
    print(out.groupby("Task")["Mean_R2_Quadratic"].mean().sort_values().round(3).to_string())
    print()
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
