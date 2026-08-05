#!/usr/bin/env python3
"""k=3 sensitivity check (CPU-only). Every FDR/dose-response result in this
study excludes Interval=3 (boundary every 3rd position) by the same
convention used throughout the report -- k=3 is the interval that most often
sits on a task's alphabet/modulus period and produces degenerate boundary
placements for some axis values. This script reruns the FDR correction
(main + reset-variant families) and the dose-response robustness fits with
k=3 INCLUDED, side by side with the standard (excluded) numbers, and reports
whether any headline conclusion -- quadratic-vs-linear win rate, the AIC/BIC/
CV split, the interior-vertex conclusion, or which cells survive FDR --
changes materially.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tasks.common.compute_diagnostics import select_canonical_files  # noqa: E402
from tasks.common.compute_fdr_correction import (  # noqa: E402
    MAIN_TASKS, RESET_TASKS, collect_family, benjamini_hochberg,
)
from tasks.common.compute_dose_response_robustness import (  # noqa: E402
    build_curve, fit_and_score, loo_density_cv, quadratic, linear,
    vertex_of_quadratic,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results_clean" / "_k3_sensitivity"


def fdr_side_by_side(task_list, family_name):
    excl = collect_family(task_list, exclude_k3=True)
    incl = collect_family(task_list, exclude_k3=False)
    for df in (excl, incl):
        df["Sig_Uncorrected_2sigma"] = df["Sigma"].abs() > 2
        df["Sig_BH_q0.05"] = benjamini_hochberg(df["P_Value"].values, alpha=0.05)
    n_excl_sig = df_sig_count(excl)
    n_incl_sig = df_sig_count(incl)
    print(f"\n{family_name}: k3-excluded n={len(excl)}, BH-sig={n_excl_sig} | "
          f"k3-included n={len(incl)}, BH-sig={n_incl_sig}")
    excl.to_csv(OUT_DIR / f"{family_name}_k3_excluded.csv", index=False)
    incl.to_csv(OUT_DIR / f"{family_name}_k3_included.csv", index=False)
    return excl, incl


def df_sig_count(df):
    return int(df["Sig_BH_q0.05"].sum())


def dose_response_side_by_side():
    rows_excl, rows_incl = [], []
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
                    for exclude_k3, sink in [(True, rows_excl), (False, rows_incl)]:
                        x, y = build_curve(df, length, axis_val, axis_col, exclude_k3=exclude_k3)
                        if len(x) < 5:
                            continue
                        scores = fit_and_score(x, y)
                        if scores is None:
                            continue
                        row = {"Task": task, "Model": model, "Seq_Length": length,
                               "Axis_Val": axis_val, "N_Densities": len(x)}
                        valid_aic = {k: v["aic"] for k, v in scores.items() if v is not None}
                        row["AIC_Winner"] = min(valid_aic, key=valid_aic.get) if valid_aic else None
                        valid_bic = {k: v["bic"] for k, v in scores.items() if v is not None}
                        row["BIC_Winner"] = min(valid_bic, key=valid_bic.get) if valid_bic else None
                        cv_quad = loo_density_cv(x, y, quadratic, [0.0, 0.0, 0.5])
                        cv_lin = loo_density_cv(x, y, linear, [0.0, 0.5])
                        row["CV_Favors_Quadratic"] = (cv_quad is not None and cv_lin is not None and cv_quad < cv_lin)
                        if scores.get("quadratic") is not None:
                            a, b, c = scores["quadratic"]["params"]
                            vertex = vertex_of_quadratic(a, b, c)
                            row["Vertex"] = vertex
                            row["Interior_Vertex"] = (vertex is not None and 0 <= vertex <= 1)
                        sink.append(row)

    excl_df, incl_df = pd.DataFrame(rows_excl), pd.DataFrame(rows_incl)
    excl_df.to_csv(OUT_DIR / "dose_response_k3_excluded.csv", index=False)
    incl_df.to_csv(OUT_DIR / "dose_response_k3_included.csv", index=False)

    def summarize(df, label):
        n = len(df)
        aic_quad = (df["AIC_Winner"] == "quadratic").sum()
        bic_quad = (df["BIC_Winner"] == "quadratic").sum()
        cv_quad = df["CV_Favors_Quadratic"].sum()
        interior = df["Interior_Vertex"].sum()
        print(f"  {label}: n={n}  AIC-quad={aic_quad}/{n} ({aic_quad/n:.1%})  "
              f"BIC-quad={bic_quad}/{n} ({bic_quad/n:.1%})  "
              f"CV-quad={cv_quad}/{n} ({cv_quad/n:.1%})  "
              f"interior-vertex={interior}/{n} ({interior/n:.1%})")

    print("\nDose-response model-selection, k3-excluded vs k3-included:")
    summarize(excl_df, "k3-excluded (standard)")
    summarize(incl_df, "k3-included")
    return excl_df, incl_df


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("=== FDR: main_grid ===")
    fdr_side_by_side(MAIN_TASKS, "main_grid")
    print("\n=== FDR: reset_variant_grid ===")
    fdr_side_by_side(RESET_TASKS, "reset_variant_grid")

    print("\n=== Dose-response robustness ===")
    dose_response_side_by_side()

    print(f"\nAll side-by-side CSVs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
