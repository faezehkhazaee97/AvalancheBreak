#!/usr/bin/env python3
"""Gemini per-length x per-density breakdown tables, formatted for a paper
appendix (CPU-only, no model calls -- reuses the existing pooled 3-trial
Gemini results). Reviewer follow-up: existing Gemini section only reports
9 pooled (Task, Base-vs-Bounds) numbers; this expands each task to a full
Seq_Length x Interval grid (mean TAR, SEM, n) matching the level of detail
already given for the 5 documented open-weight models.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = REPO_ROOT / "results_clean" / "gemini_pilot" / "gemini-flash-latest_POOLED_3trials.csv"
OUT_DIR = REPO_ROOT / "results_clean" / "_gemini_breakdowns"


def sem(x):
    x = np.asarray(x)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def main():
    df = pd.read_csv(IN_CSV)
    df = df[df.Interval != 3]  # standard k=3 exclusion convention
    OUT_DIR.mkdir(exist_ok=True)

    md_lines = ["# Gemini (gemini-flash-latest) per-length x per-density breakdown\n",
                f"Source: `{IN_CSV.relative_to(REPO_ROOT)}`, n=3 trials/cell, k=3 excluded "
                "(standard convention).\n"]

    for task, tdf in df.groupby("Task"):
        label = "Density (1/k)" if tdf["Interval"].max() > 0 else "Interval"
        rows = []
        for (length, interval), sub in tdf.groupby(["Seq_Length", "Interval"]):
            rows.append({
                "Seq_Length": length,
                "Method": "Base" if interval == 0 else "Cellular_Bounds",
                "Interval": interval,
                "Mean_TAR": sub["TAR"].mean(),
                "SEM_TAR": sem(sub["TAR"].values),
                "Mean_SAR": sub["SAR"].mean(),
                "SEM_SAR": sem(sub["SAR"].values),
                "N": len(sub),
            })
        out_df = pd.DataFrame(rows).sort_values(["Seq_Length", "Interval"])
        out_csv = OUT_DIR / f"{task}_breakdown.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"{task}: wrote {len(out_df)} rows to {out_csv}")

        md_lines.append(f"\n## {task}\n")
        md_lines.append("| Seq_Length | Method | k | Mean TAR | SEM | Mean SAR | SEM | n |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in out_df.iterrows():
            md_lines.append(
                f"| {int(r.Seq_Length)} | {r.Method} | {int(r.Interval)} | "
                f"{r.Mean_TAR:.3f} | {r.SEM_TAR:.3f} | {r.Mean_SAR:.3f} | {r.SEM_SAR:.3f} | "
                f"{int(r.N)} |"
            )

    md_path = OUT_DIR / "gemini_breakdowns_appendix.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"\nWrote appendix-formatted markdown to {md_path}")


if __name__ == "__main__":
    main()
