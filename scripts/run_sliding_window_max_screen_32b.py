#!/usr/bin/env python3
"""Same sliding_window_max Base-only screen as run_sliding_window_max_screen.py,
on Qwen2.5-32B-Instruct instead of 7B -- checks whether the alphabet=2
weakness (never clearing the naive-baseline bar) is a 7B-specific ceiling
effect or a property of the task itself at that alphabet size.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.sliding_window_max import engine
from tasks.common.resumable_runner import run_screen

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
OUTPUT_CSV = REPO_ROOT / "results" / "external_tasks" / "sliding_window_max" / f"{MODEL_TAG}_screen.csv"

run_screen(engine, MODEL_ID, str(OUTPUT_CSV), trials_per_cell=10)
