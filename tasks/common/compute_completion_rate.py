#!/usr/bin/env python3
"""Q7 (cheap half): does boundary density compound early termination?

Completion Rate (CR) = fraction of trials whose output is at least as long as
expected (the `Complete` diagnostic column from compute_diagnostics.py),
broken out by Interval (k) instead of pooled Base-vs-Bounds. Reads the
per-model `_rows.csv` files that compute_diagnostics.py writes -- restricted
to the 5 models documented in overleaf/main.tex (see DOCUMENTED_MODELS in
compute_diagnostics.py; OLMo-2-1124-7B-Instruct and Qwen2.5-72B-Instruct rows
were previously leaking into this analysis and have been moved to
results_clean/_diagnostics/_excluded_undocumented_models/).

Pooling: CR per (Task, Model, Interval) cell first, then the "pooled" table
is an unweighted mean of those cells across all Task x Model combinations at
a given Interval -- matches the convention used in the original writeup
(every task/model contributes equally, not weighted by row count).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tasks.common.compute_diagnostics import CLEAN_TASKS, DOCUMENTED_MODELS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "results_clean" / "_diagnostics"


def density_of(k):
    return 0.0 if k == 0 else 1.0 / k


def main():
    rows_files = sorted(DIAG.glob("*__rows.csv"))
    cells = []
    for f in rows_files:
        stem = f.stem[: -len("__rows")]
        task, model = stem.split("__", 1)
        if task not in CLEAN_TASKS or model not in DOCUMENTED_MODELS:
            continue
        df = pd.read_csv(f)
        df = df[df["Interval"] != 3]
        for k, sub in df.groupby("Interval"):
            cells.append({
                "Task": task, "Model": model, "Interval": k,
                "Method": "Base" if k == 0 else "Cellular_Bounds",
                "N": len(sub), "CR": sub["Complete"].mean(),
            })
    cell_df = pd.DataFrame(cells)
    out_csv = DIAG / "completion_rate_by_density.csv"
    cell_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(cell_df)} rows to {out_csv} ({cell_df['Task'].nunique()} tasks x "
          f"{cell_df['Model'].nunique()} models)")

    print("\nPooled (unweighted mean across Task x Model cells) by Interval:")
    pooled = cell_df.groupby("Interval")["CR"].mean().sort_index()
    for k, cr in pooled.items():
        label = "Base" if k == 0 else str(int(k))
        print(f"  k={label:>4}  density={density_of(k):.3f}  pooled CR={cr:.3f}")

    print("\nPer-task correlation between density and CR (Cellular_Bounds cells only):")
    corr_rows = []
    for task, sub in cell_df[cell_df.Interval != 0].groupby("Task"):
        piv = sub.pivot_table(index="Model", columns="Interval", values="CR")
        dens = [density_of(k) for k in piv.columns]
        # flatten (model, interval) pairs for a single pooled correlation per task
        xs, ys = [], []
        for _, row in piv.iterrows():
            for k, cr in zip(piv.columns, row.values):
                if not np.isnan(cr):
                    xs.append(density_of(k))
                    ys.append(cr)
        corr = np.corrcoef(xs, ys)[0, 1] if len(xs) > 2 else float("nan")
        corr_rows.append({"Task": task, "Corr": corr})
    corr_df = pd.DataFrame(corr_rows).sort_values("Corr")
    for _, r in corr_df.iterrows():
        print(f"  {r['Task']:<25} corr={r['Corr']:+.3f}")

    return cell_df, pooled, corr_df


if __name__ == "__main__":
    main()
