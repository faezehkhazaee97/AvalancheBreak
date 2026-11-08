"""Sanity gate for a newly added model, run at the end of its smoke job.

Reads the verify-mode CSVs (1 trial/cell) for cyclic_shift and copy_previous
and exits non-zero if the outputs look broken, so the dependent full-grid
array (submitted with --dependency=afterok) never starts on a bad setup.

Fails on:
  - reasoning text in the raw output (thinking mode not actually disabled)
  - low completion rate (model stopping early / answering in prose)
  - near-zero accuracy (e.g. weights that didn't load, wrong model class)

Usage: python scripts/validate_new_model_smoke.py <model_tag>
"""
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TASKS = ["cyclic_shift", "copy_previous"]
MIN_COMPLETION = 0.6
MIN_MEAN_TAR = 0.25
THINK = re.compile(r"</?think>|<\|channel>|<channel\|>|\bthought\b", re.I)

tag = sys.argv[1]
ok = True
for task in TASKS:
    path = REPO / "results_clean" / task / "verify" / f"{tag}.csv"
    if not path.exists():
        print(f"FAIL {task}: {path} missing")
        ok = False
        continue
    df = pd.read_csv(path, keep_default_na=False)
    raw = df["Raw_Model_Output"].astype(str)
    think = raw.str.contains(THINK).mean()
    completion = (df["LLM_Output"].astype(str).str.len() >= df["Ground_Truth"].astype(str).str.len()).mean()
    tar = df["TAR"].mean()
    by_method = df.groupby("Method")["TAR"].mean().round(3).to_dict()
    print(f"{task}: rows={len(df)} think_rate={think:.2f} completion={completion:.2f} "
          f"mean_TAR={tar:.3f} by_method={by_method}")
    print("  sample raw outputs:", [r[:60] for r in raw.head(3)])
    if think > 0:
        print(f"FAIL {task}: reasoning text found in {think:.0%} of outputs")
        ok = False
    if completion < MIN_COMPLETION:
        print(f"FAIL {task}: completion {completion:.2f} < {MIN_COMPLETION}")
        ok = False
    if tar < MIN_MEAN_TAR:
        print(f"FAIL {task}: mean TAR {tar:.3f} < {MIN_MEAN_TAR}")
        ok = False

print("SMOKE PASSED" if ok else "SMOKE FAILED")
sys.exit(0 if ok else 1)
