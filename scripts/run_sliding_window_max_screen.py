#!/usr/bin/env python3
"""Base-only competence screen for sliding_window_max (second replacement
candidate for running_max, after running_majority failed its screen).
Same convention: 10 trials/cell, Base only, check TAR clears chance
robustly before committing to a full grid.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.sliding_window_max import engine
from tasks.common.resumable_runner import run_screen

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
OUTPUT_CSV = REPO_ROOT / "results" / "external_tasks" / "sliding_window_max" / f"{MODEL_TAG}_screen.csv"

run_screen(engine, MODEL_ID, str(OUTPUT_CSV), trials_per_cell=10)
