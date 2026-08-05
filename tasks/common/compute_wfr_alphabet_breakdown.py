#!/usr/bin/env python3
"""Appendix F, by-alphabet-size breakdown: the pooled well-formedness table
(all_diagnostics_summary.csv) collapses across Alphabet_Size/Modulus -- this
reproduces the same metric set (TAR, SAR, VVR, CR, WFR, NEO, TAR|WF) broken
out by that axis too, for the 6 tasks that have one. Reads the clean,
post-decontamination _rows.csv files (5 documented models only).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tasks.common.compute_diagnostics import CLEAN_TASKS, DOCUMENTED_MODELS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "results_clean" / "_diagnostics"

AXIS_TASKS = {
    "cyclic_shift": "Alphabet_Size", "character_substitution": "Alphabet_Size",
    "random_permutation": "Alphabet_Size", "conditional_shift": "Alphabet_Size",
    "composed_permutation": "Alphabet_Size", "running_max": "Alphabet_Size",
    "running_sum_local": "Modulus",
}


def main():
    rows = []
    for f in sorted(DIAG.glob("*__rows.csv")):
        stem = f.stem[: -len("__rows")]
        task, model = stem.split("__", 1)
        if task not in CLEAN_TASKS or model not in DOCUMENTED_MODELS or task not in AXIS_TASKS:
            continue
        axis_col = AXIS_TASKS[task]
        df = pd.read_csv(f)
        df = df[df["Interval"] != 3]
        if axis_col not in df.columns:
            continue
        for (method, axis_val), sub in df.groupby(["Method", axis_col]):
            wf = sub[sub["Well_Formed"]]
            rows.append({
                "Task": task, "Model": model, "Method": method, axis_col: axis_val,
                "N_trials": len(sub),
                "TAR": sub["TAR"].mean(), "SAR": sub["SAR"].mean(),
                "Vocab_Valid_Rate": sub["Vocab_Valid"].mean(),
                "Completion_Rate": sub["Complete"].mean(),
                "Well_Formed_Rate": sub["Well_Formed"].mean(),
                "Normalized_Error_Onset": sub["Normalized_Error_Onset"].mean(),
                "TAR_given_WellFormed": wf["TAR"].mean() if len(wf) else float("nan"),
            })
    out = pd.DataFrame(rows)
    out_path = DIAG / "wfr_by_alphabet_size_breakdown.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")
    print(out.head(15).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
