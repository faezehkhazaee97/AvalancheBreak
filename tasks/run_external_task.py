"""CLI entry point for the external-tasks suite (procbench_substitute,
string_copy, flip_flop_high, flip_flop_low). Results go under
results/external_tasks/, distinct from the main study's results_clean/.

Usage:
    python -m tasks.run_external_task <task_name> <model_id> screen [trials_per_cell]
    python -m tasks.run_external_task <task_name> <model_id> grid [trials_per_cell]

screen: Base-only (k=0) pass across all lengths, default 10 trials/cell.
grid:   full Base+Cellular_Bounds grid matching the main study's
        STANDARD_LENGTHS/STANDARD_INTERVALS, default 20 trials/cell.
Both modes append incrementally and resume from an existing output CSV.
"""

import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.common import resumable_runner  # noqa: E402

EXTERNAL_TASK_NAMES = ["procbench_substitute", "string_copy", "flip_flop_high", "flip_flop_low"]


def main():
    if len(sys.argv) < 4:
        print("Usage: python -m tasks.run_external_task <task_name> <model_id> <screen|grid> [trials_per_cell]")
        print(f"task_name options: {EXTERNAL_TASK_NAMES}")
        sys.exit(1)

    task_name, model_id, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    if task_name not in EXTERNAL_TASK_NAMES:
        print(f"Unknown external task '{task_name}'. Options: {EXTERNAL_TASK_NAMES}")
        sys.exit(1)
    if mode not in ("screen", "grid"):
        print(f"Unknown mode '{mode}'. Options: screen, grid")
        sys.exit(1)

    engine = importlib.import_module(f"tasks.{task_name}.engine")
    model_tag = model_id.split("/")[-1]
    output_csv = os.path.join(REPO_ROOT, "results", "external_tasks", task_name, f"{model_tag}_{mode}.csv")

    if mode == "screen":
        trials_per_cell = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        resumable_runner.run_screen(engine, model_id, output_csv, trials_per_cell=trials_per_cell)
    else:
        trials_per_cell = int(sys.argv[4]) if len(sys.argv) > 4 else 20
        resumable_runner.run_grid_resumable(engine, model_id, output_csv, trials_per_cell=trials_per_cell)


if __name__ == "__main__":
    main()
