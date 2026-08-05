#!/usr/bin/env python3
"""Benjamini-Hochberg FDR correction across every Base-vs-Cellular_Bounds
significance test in the study, at the same per-cell granularity already
used throughout the report (per task x model x length x axis_val cell,
pooled across trials and boundary densities within that cell, k=3
excluded -- matching the convention used everywhere else in this report).

Reviewer's concern (Q2): "Given the large number of cells and significance
tests, did you consider FDR control or a hierarchical testing framework?"

Corrects within logical families separately (correcting across totally
different analyses in one pool isn't statistically meaningful):
  - "main_grid": every per-cell test across the 9 core tasks' full grids
    (the family the reviewer's comment is really about -- by far the
    largest number of tests)
  - "reset_variant_grid": the 5 causal-ablation reset-variant tasks' grids
  - "causal_ablation_pooled": the 10 pooled Original/Reset tests already
    in the causal-ablation table
  - "gemini_pooled": the 9 pooled tests already in the Gemini table
  - "clrs_pooled": the 2 pooled tests already in the CLRS-Text table
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tasks.common.compute_diagnostics import select_canonical_files, DOCUMENTED_MODELS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results_clean"

MAIN_TASKS = [
    "cyclic_shift", "character_substitution", "random_permutation",
    "conditional_shift", "composed_permutation", "copy_previous", "running_max",
    "running_xor", "running_sum_local",
]
RESET_TASKS = [
    "running_xor_reset", "copy_previous_reset", "running_sum_local_reset",
    "reverse_xor_reset", "running_max_reset",
]

# DOCUMENTED_MODELS (imported above) restricts select_canonical_files() to the
# 5 models overleaf/main.tex actually describes -- see compute_diagnostics.py
# for why (OLMo/Qwen2.5-72B leftover-run exclusion).


def two_tailed_p(sigma):
    return 2 * (1 - stats.norm.cdf(abs(sigma)))


def benjamini_hochberg(pvals, alpha=0.05):
    """Returns boolean array: True where the test survives BH correction
    at the given FDR level. Standard BH step-up procedure."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresholds
    if not below.any():
        reject = np.zeros(n, dtype=bool)
    else:
        max_i = np.where(below)[0].max()
        reject = np.zeros(n, dtype=bool)
        reject[: max_i + 1] = True
    result = np.zeros(n, dtype=bool)
    result[order] = reject
    return result


def per_cell_tests(task, csv_paths, exclude_k3=True):
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    key_cols = [c for c in ["Method", "Interval", "Seq_Length", "Alphabet_Size", "Modulus", "Trial"] if c in df.columns]
    df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    if exclude_k3 and "Interval" in df.columns:
        df = df[df["Interval"] != 3].reset_index(drop=True)

    axis_col = "Alphabet_Size" if "Alphabet_Size" in df.columns else ("Modulus" if "Modulus" in df.columns else None)
    group_cols = ["Seq_Length"] + ([axis_col] if axis_col else [])

    rows = []
    for key, sub in df.groupby(group_cols):
        base = sub[sub.Method == "Base"].TAR
        bnd = sub[sub.Method == "Cellular_Bounds"].TAR
        if len(base) < 2 or len(bnd) < 2:
            continue
        delta = bnd.mean() - base.mean()
        sem = np.sqrt(base.sem() ** 2 + bnd.sem() ** 2)
        if sem == 0 or np.isnan(sem):
            continue
        sigma = delta / sem
        p = two_tailed_p(sigma)
        key_vals = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_vals))
        row.update({"Task": task, "Delta": delta, "Sigma": sigma, "P_Value": p,
                     "N_Base": len(base), "N_Bounds": len(bnd)})
        rows.append(row)
    return rows


def collect_family(task_list, exclude_k3=True):
    rows = []
    for task in task_list:
        model_files = select_canonical_files(task)
        for model, paths in model_files.items():
            cell_rows = per_cell_tests(task, paths, exclude_k3=exclude_k3)
            for r in cell_rows:
                r["Model"] = model
            rows.extend(cell_rows)
    return pd.DataFrame(rows)


def report_family(name, df, alpha_levels=(0.05, 0.10)):
    print(f"\n{'='*70}\nFamily: {name}  (n={len(df)} tests)\n{'='*70}")
    if len(df) == 0:
        print("  (no tests)")
        return df
    df = df.copy()
    df["Sig_Uncorrected_2sigma"] = df["Sigma"].abs() > 2
    df["Sig_Uncorrected_3sigma"] = df["Sigma"].abs() > 3
    for alpha in alpha_levels:
        df[f"Sig_BH_q{alpha}"] = benjamini_hochberg(df["P_Value"].values, alpha=alpha)

    n_unc_2s = df["Sig_Uncorrected_2sigma"].sum()
    n_unc_3s = df["Sig_Uncorrected_3sigma"].sum()
    print(f"  Uncorrected: {n_unc_2s}/{len(df)} significant at >2sigma, {n_unc_3s}/{len(df)} at >3sigma")
    for alpha in alpha_levels:
        n_bh = df[f"Sig_BH_q{alpha}"].sum()
        print(f"  BH-corrected at FDR={alpha}: {n_bh}/{len(df)} remain significant")

    lost = df[df["Sig_Uncorrected_2sigma"] & ~df[f"Sig_BH_q{alpha_levels[0]}"]]
    if len(lost):
        print(f"\n  Tests significant at >2sigma uncorrected but LOSE significance at BH FDR={alpha_levels[0]}:")
        cols = [c for c in ["Task", "Model", "Seq_Length", "Alphabet_Size", "Modulus", "Delta", "Sigma", "P_Value"] if c in lost.columns]
        print(lost[cols].sort_values("Sigma", key=abs, ascending=False).to_string(index=False))
    else:
        print(f"\n  No test loses >2sigma significance under BH FDR={alpha_levels[0]} correction.")
    return df


def main():
    out_dir = RESULTS / "_fdr_correction"
    out_dir.mkdir(exist_ok=True)

    print("Collecting main-grid per-cell tests (9 core tasks, all models)...")
    main_df = collect_family(MAIN_TASKS)
    main_df = report_family("main_grid", main_df)
    main_df.to_csv(out_dir / "main_grid_fdr.csv", index=False)

    print("\nCollecting reset-variant per-cell tests (5 reset tasks)...")
    reset_df = collect_family(RESET_TASKS)
    reset_df = report_family("reset_variant_grid", reset_df)
    reset_df.to_csv(out_dir / "reset_variant_grid_fdr.csv", index=False)

    # Causal ablation pooled (10 tests: 5 original + 5 reset, Qwen-7B only, pooled across all lengths)
    print("\nCollecting causal-ablation pooled tests (10 tests, already-pooled numbers)...")
    causal_rows = []
    CAUSAL_TASKS = {
        "running_xor": (RESULTS / "running_xor/grid/Qwen2.5-7B-Instruct.csv", RESULTS / "running_xor_reset/grid/Qwen2.5-7B-Instruct.csv"),
        "copy_previous": (RESULTS / "copy_previous/grid/Qwen2.5-7B-Instruct_POOLED_40trials.csv", RESULTS / "copy_previous_reset/grid/Qwen2.5-7B-Instruct.csv"),
        "running_sum_local": (RESULTS / "running_sum_local/grid/Qwen2.5-7B-Instruct.csv", RESULTS / "running_sum_local_reset/grid/Qwen2.5-7B-Instruct.csv"),
        "reverse_xor": (RESULTS / "reverse_xor/verify/Qwen2.5-7B-Instruct_FULL_GRID_40trials.csv", RESULTS / "reverse_xor_reset/grid/Qwen2.5-7B-Instruct.csv"),
        "running_max": (RESULTS / "running_max/grid/Qwen2.5-7B-Instruct_POOLED_40trials.csv", RESULTS / "running_max_reset/grid/Qwen2.5-7B-Instruct.csv"),
    }
    def pooled_test(path, label, task):
        df = pd.read_csv(path)
        df = df[df.Interval != 3]
        base = df[df.Method == "Base"].TAR
        bnd = df[df.Method == "Cellular_Bounds"].TAR
        delta = bnd.mean() - base.mean()
        sem = np.sqrt(base.sem() ** 2 + bnd.sem() ** 2)
        sigma = delta / sem
        return {"Task": task, "Variant": label, "Delta": delta, "Sigma": sigma, "P_Value": two_tailed_p(sigma)}
    for task, (orig_p, reset_p) in CAUSAL_TASKS.items():
        causal_rows.append(pooled_test(orig_p, "Original", task))
        causal_rows.append(pooled_test(reset_p, "Reset", task))
    causal_df = pd.DataFrame(causal_rows)
    causal_df = report_family("causal_ablation_pooled", causal_df)
    causal_df.to_csv(out_dir / "causal_ablation_fdr.csv", index=False)

    # Gemini pooled (9 tests)
    print("\nCollecting Gemini pooled tests (9 tests)...")
    gemini_path = RESULTS / "gemini_pilot" / "gemini-flash-latest_POOLED_3trials.csv"
    gemini_rows = []
    if gemini_path.exists():
        gdf = pd.read_csv(gemini_path)
        gdf = gdf[gdf.Interval != 3]
        for task in gdf.Task.unique():
            sub = gdf[gdf.Task == task]
            base = sub[sub.Method == "Base"].TAR
            bnd = sub[sub.Method == "Cellular_Bounds"].TAR
            delta = bnd.mean() - base.mean()
            sem = np.sqrt(base.sem() ** 2 + bnd.sem() ** 2)
            sigma = delta / sem
            gemini_rows.append({"Task": task, "Delta": delta, "Sigma": sigma, "P_Value": two_tailed_p(sigma)})
    gemini_df = pd.DataFrame(gemini_rows)
    gemini_df = report_family("gemini_pooled", gemini_df)
    gemini_df.to_csv(out_dir / "gemini_fdr.csv", index=False)

    # CLRS pooled (2 tests)
    print("\nCollecting CLRS-Text pooled tests (2 tests)...")
    clrs_rows = []
    for task in ["clrs_minimum", "clrs_kadane"]:
        p = RESULTS / task / "grid" / "Qwen2.5-7B-Instruct.csv"
        if p.exists():
            clrs_rows.append(pooled_test(p, "n/a", task))
    clrs_df = pd.DataFrame(clrs_rows)
    clrs_df = report_family("clrs_pooled", clrs_df)
    clrs_df.to_csv(out_dir / "clrs_fdr.csv", index=False)

    print(f"\n\nAll family CSVs written to {out_dir}/")


if __name__ == "__main__":
    main()
