"""Extends a task's grid to N=128 for a single model, without touching the
existing standard-grid (N<=64) production data. Used for the Qwen-32B-only
main-paper figure, which needs cyclic_shift's already-existing N=128/256
points matched by comparable N=128 points on running_xor, running_sum_local,
copy_previous, and sliding_window_max (all of which only go to N=64 in the
5-model standard grid).

Writes to results_clean/<task_name>/grid/<model_tag>_N128.csv -- a separate
file from the standard grid, so it's additive (figure scripts concatenate it
in for Qwen-32B) rather than a schema change to the shared 5-model grid.

Usage: python -m tasks.run_task_n128 <task_name> <model_id> [trials_per_cell]
"""

import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.common import runner  # noqa: E402

TASK_NAMES = [
    "running_xor", "running_sum_local", "copy_previous", "sliding_window_max",
    "random_permutation", "conditional_shift",
]


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m tasks.run_task_n128 <task_name> <model_id> [trials_per_cell]")
        print(f"task_name options: {TASK_NAMES}")
        sys.exit(1)

    task_name, model_id = sys.argv[1], sys.argv[2]
    trials_per_cell = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    if task_name not in TASK_NAMES:
        print(f"Unknown task '{task_name}'. Options: {TASK_NAMES}")
        sys.exit(1)

    engine = importlib.import_module(f"tasks.{task_name}.engine")
    engine.LENGTHS = [128]  # override -- only generate the new N=128 cell

    model_tag = model_id.split("/")[-1]
    output_csv = os.path.join(REPO_ROOT, "results_clean", task_name, "grid", f"{model_tag}_N128.csv")

    runner.run(engine, model_id, "grid", output_csv, trials_per_cell=trials_per_cell)


if __name__ == "__main__":
    main()
