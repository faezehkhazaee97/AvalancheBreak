#!/usr/bin/env python3
"""Concatenate the Gemini 3.6 Flash trial files into one pooled file, the shape the analysis
scripts read for Gemini (same columns as the trial files; one row per generation).

Input : results_clean/gemini_pilot/gemini-3.6-flash_full_grid_trial*.csv
Output: results_clean/gemini_pilot/gemini-3.6-flash_POOLED.csv
"""
import glob
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
D = REPO / "results_clean" / "gemini_pilot"
fs = sorted(glob.glob(str(D / "gemini-3.6-flash_full_grid_trial*.csv")))
d = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in fs], ignore_index=True)
n = len(d)
d = d.drop_duplicates(["Task", "Method", "Interval", "Seq_Length", "Axis_Val", "Trial"])
assert set(d.Model_Resolved.unique()) == {"gemini-3.6-flash"}, d.Model_Resolved.unique()
err = (d.Error != "").sum()
d.to_csv(D / "gemini-3.6-flash_POOLED.csv", index=False)
print(f"{len(fs)} trial files, {n} rows ({n - len(d)} duplicates dropped), {err} API errors, "
      f"{d.Task.nunique()} tasks, trials {sorted(d.Trial.unique())}")
print(d.groupby("Task").size().to_string())
