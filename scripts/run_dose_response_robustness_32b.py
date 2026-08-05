#!/usr/bin/env python3
"""Extends the existing dose-response robustness machinery (AIC/BIC model
selection, leave-one-density-out CV, bootstrapped vertex CI) to:
  (1) Qwen2.5-32B-Instruct only, matching the main paper's single-model story
  (2) sliding_window_max, which wasn't in the original MAIN_TASKS list
  (3) N=128, merged in from the *_N128.csv extension files for tasks that
      only went to N=64 in the standard 5-model grid (cyclic_shift already
      natively covers up to N=256, so it's untouched)

Reuses build_curve / fit_and_score / vertex_of_quadratic / bootstrap_vertex_ci
unchanged from compute_dose_response_robustness.py so results stay directly
comparable to the existing 7-model table already in the report.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.compute_diagnostics import select_canonical_files
from tasks.common.compute_dose_response_robustness import (
    build_curve, fit_and_score, quadratic, linear, vertex_of_quadratic,
    bootstrap_vertex_ci, loo_density_cv,
)

MODEL = "Qwen2.5-32B-Instruct"
TASKS = ["cyclic_shift", "random_permutation", "conditional_shift",
         "copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]

N128_EXTENSION_TASKS = {"copy_previous", "running_xor", "running_sum_local",
                         "sliding_window_max", "random_permutation", "conditional_shift"}


def load_task_df(task):
    frames = []
    if task in N128_EXTENSION_TASKS:
        grid_dir = REPO_ROOT / "results_clean" / task / "grid"
        for p in grid_dir.glob(f"{MODEL}*.csv"):
            frames.append(pd.read_csv(p))
    else:
        files = select_canonical_files(task)
        if MODEL not in files:
            return None
        for p in files[MODEL]:
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    key_cols = [c for c in ["Method", "Interval", "Seq_Length", "Alphabet_Size", "Modulus", "Trial"] if c in df.columns]
    df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    return df


all_rows = []
for task in TASKS:
    df = load_task_df(task)
    if df is None:
        print(f"{task}: no data for {MODEL}")
        continue
    axis_col = "Alphabet_Size" if "Alphabet_Size" in df.columns else ("Modulus" if "Modulus" in df.columns else None)
    axis_values = sorted(df[axis_col].dropna().unique()) if axis_col else [None]
    lengths = sorted(df.Seq_Length.unique())

    for length in lengths:
        for axis_val in axis_values:
            x, y = build_curve(df, length, axis_val, axis_col)
            if len(x) < 5:
                continue
            scores = fit_and_score(x, y)
            if scores is None:
                continue

            row = {"Task": task, "Model": MODEL, "Seq_Length": length, "Axis_Val": axis_val, "N_Densities": len(x)}
            for name in ["linear", "quadratic", "exponential"]:
                r = scores.get(name)
                if r is not None:
                    row[f"{name}_R2"] = 1 - r["rss"] / max(np.sum((y - y.mean()) ** 2), 1e-12)
                    row[f"{name}_AIC"] = r["aic"]

            valid = {k: v["aic"] for k, v in scores.items() if v is not None}
            row["AIC_Winner"] = min(valid, key=valid.get) if valid else None

            cv_quad = loo_density_cv(x, y, quadratic, [0.0, 0.0, 0.5])
            cv_lin = loo_density_cv(x, y, linear, [0.0, 0.5])
            row["CV_Favors_Quadratic"] = (cv_quad is not None and cv_lin is not None and cv_quad < cv_lin)

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
out_dir = REPO_ROOT / "results_clean" / "_dose_response_robustness"
out_dir.mkdir(exist_ok=True)
out.to_csv(out_dir / "dose_response_robustness_32b.csv", index=False)
print(f"Wrote {len(out)} curves to {out_dir}/dose_response_robustness_32b.csv")

print("\n=== Per-task summary: mean quadratic R^2, AIC-winner rate, vertex-with-CI at each length ===")
for task in TASKS:
    sub = out[out.Task == task]
    if len(sub) == 0:
        continue
    quad_r2 = sub["quadratic_R2"].mean()
    quad_wins = (sub.AIC_Winner == "quadratic").sum()
    print(f"\n--- {task} (mean quadratic R^2={quad_r2:.3f}, quadratic wins AIC in {quad_wins}/{len(sub)}) ---")
    for _, r in sub.sort_values(["Axis_Val", "Seq_Length"]).iterrows():
        v = r.get("Vertex")
        lo, hi = r.get("Vertex_CI_Low"), r.get("Vertex_CI_High")
        vstr = f"vertex={v:.3f} [{lo:.3f},{hi:.3f}]" if pd.notna(v) and pd.notna(lo) else (f"vertex={v:.3f} [CI failed]" if pd.notna(v) else "no vertex")
        print(f"  axis={r.Axis_Val} N={int(r.Seq_Length):>4}  R2={r.quadratic_R2:.3f}  CV_favors_quad={r.CV_Favors_Quadratic}  {vstr}")
