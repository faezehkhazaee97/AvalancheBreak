#!/usr/bin/env python3
"""Compute the new diagnostic metrics (Vocab_Valid, Complete, Non_Degenerate,
Well_Formed, Normalized_Error_Onset) over every stored trial for all 9 core
tasks, now that results_clean/<task>/grid/ is clean for all of them --
running_xor and running_sum_local were reran through tasks.common.runner for
Qwen-7B/14B/32B/Mistral-7B on 2026-07-18 (single 40-trial batch each,
job IDs 36209098-36209105) specifically to close this gap; their Qwen/Mistral
history before that lived in results_clean/<task>/verify/ as dozens of
overlapping, differently-named experimental fragments (prompt-variant tests,
topups, "EVEN_POOL" rebalanced files that dropped LLM_Output) that were never
safe to reconstruct from.

Canonical-file selection per model: prefer "<model>_POOLED_40trials.csv" if
present (the deduplicated 2-batch superset), else the bare "<model>.csv".
Raw "_trials0-19"/"_trials20-39" fragments are always skipped -- they're
subsets of the POOLED file, keeping them would double-count trials.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tasks.common import metrics as metrics_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results_clean"

CLEAN_TASKS = [
    "cyclic_shift", "character_substitution", "random_permutation",
    "conditional_shift", "composed_permutation", "copy_previous", "running_max",
    "running_xor", "running_sum_local",
]
PARTIAL_TASKS = []


def select_canonical_files(task):
    """Group all non-aggregated (LLM_Output-preserving) grid CSVs by model.
    "_POOLED_40trials" files are aggregate-only (Method/Interval/Seq_Length/
    TAR/SAR, no LLM_Output) and are always skipped here -- the raw fragments
    they were built from ("<model>.csv" and/or "_trials0-19"/"_trials20-39")
    are concatenated instead so LLM_Output survives. Coverage can be partial
    per model (e.g. only one 20-trial fragment retains raw output for some
    cyclic_shift models) -- N_trials in the summary makes this visible."""
    grid_dir = RESULTS / task / "grid"
    if not grid_dir.exists():
        return {}
    files = sorted(grid_dir.glob("*.csv"))
    by_model = {}
    for f in files:
        name = f.stem
        if name.endswith("_POOLED_40trials"):
            continue
        model = name
        for suffix in ["_trials0-19", "_trials20-39"]:
            if model.endswith(suffix):
                model = model[: -len(suffix)]
                break
        by_model.setdefault(model, []).append(f)
    return by_model


def compute_for_file(task, csv_paths):
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    key_cols = [c for c in ["Method", "Interval", "Seq_Length", "Alphabet_Size", "Modulus", "Trial"] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    if len(df) != before:
        print(f"  dropped {before - len(df)} duplicate rows on {key_cols}")

    if "Interval" in df.columns:
        before = len(df)
        df = df[df["Interval"] != 3].reset_index(drop=True)
        if len(df) != before:
            print(f"  dropped {before - len(df)} boundary-every-3 (Interval=3) rows")

    diag_rows = [metrics_mod.row_diagnostics(task, row) for _, row in df.iterrows()]
    diag_df = pd.DataFrame(diag_rows)
    diag_df["Overlong"] = diag_df["Overlong"].astype("boolean")  # nullable: None where Raw_Model_Output unavailable
    out = pd.concat([df.reset_index(drop=True), diag_df], axis=1)
    return out


def summarize(df, group_col="Method"):
    g = df.groupby(group_col)
    summary = pd.DataFrame({
        "N_trials": g.size(),
        "TAR": g["TAR"].mean(),
        "SAR": g["SAR"].mean(),
        "Vocab_Valid_Rate": g["Vocab_Valid"].mean(),
        "Completion_Rate": g["Complete"].mean(),
        "Well_Formed_Rate": g["Well_Formed"].mean(),
        "Normalized_Error_Onset": g["Normalized_Error_Onset"].mean(),
        "Overlong_Rate": g["Overlong"].mean(),
        "Boundary_Fidelity": g["Boundary_Fidelity"].mean(),
        "N_with_raw_output": g["Overlong"].apply(lambda s: s.notna().sum()),
    })
    wf = df[df["Well_Formed"]]
    summary["TAR_given_WellFormed"] = wf.groupby(group_col)["TAR"].mean()
    summary.index.name = "Method"
    return summary.reset_index()


def summarize_all(df):
    row = {
        "Method": "All",
        "N_trials": len(df),
        "TAR": df["TAR"].mean(),
        "SAR": df["SAR"].mean(),
        "Vocab_Valid_Rate": df["Vocab_Valid"].mean(),
        "Completion_Rate": df["Complete"].mean(),
        "Well_Formed_Rate": df["Well_Formed"].mean(),
        "Normalized_Error_Onset": df["Normalized_Error_Onset"].mean(),
        "Overlong_Rate": df["Overlong"].mean(),
        "Boundary_Fidelity": df["Boundary_Fidelity"].mean(),
        "N_with_raw_output": df["Overlong"].notna().sum(),
        "TAR_given_WellFormed": df.loc[df["Well_Formed"], "TAR"].mean(),
    }
    return pd.DataFrame([row])


def main():
    out_dir = RESULTS / "_diagnostics"
    out_dir.mkdir(exist_ok=True)

    all_summaries = []
    for task in CLEAN_TASKS + PARTIAL_TASKS:
        model_files = select_canonical_files(task) if task in CLEAN_TASKS else {}
        if task in PARTIAL_TASKS:
            phi_path = RESULTS / task / "grid" / "Phi-3.5-mini-instruct.csv"
            if phi_path.exists():
                model_files = {"Phi-3.5-mini-instruct": [phi_path]}

        for model, paths in model_files.items():
            print(f"Processing {task} / {model} ({len(paths)} file(s)) ...")
            df = compute_for_file(task, paths)
            row_out = out_dir / f"{task}__{model}__rows.csv"
            df.to_csv(row_out, index=False)

            summary = summarize(df)
            summary.insert(0, "Model", model)
            summary.insert(0, "Task", task)
            all_summaries.append(summary)

            summary_overall = summarize_all(df)
            summary_overall.insert(0, "Model", model)
            summary_overall.insert(0, "Task", task)
            all_summaries.append(summary_overall)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_out = out_dir / "all_diagnostics_summary.csv"
    combined.to_csv(combined_out, index=False)
    print(f"\nWrote {combined_out} ({len(combined)} rows)")
    print(combined.to_string())


if __name__ == "__main__":
    main()
