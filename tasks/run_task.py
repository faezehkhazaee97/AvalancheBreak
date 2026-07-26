"""Generic CLI entry point for any of the 6 task engines.

Usage:
    python -m tasks.run_task <task_name> <model_id> <mode> [trials_per_cell]

task_name  : cyclic_shift | character_substitution | running_xor |
             running_sum_local | modsum | addition
mode       : verify | grid
Output goes to results_clean/<task_name>/<mode>/<model_tag>.csv
"""

import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.common import runner  # noqa: E402

TASK_NAMES = [
    "cyclic_shift", "character_substitution", "running_xor",
    "running_sum_local", "modsum", "addition",
    "random_permutation", "conditional_shift", "composed_permutation",
    "reverse_xor", "copy_previous", "running_max",
    "running_xor_reset", "running_sum_local_reset", "reverse_xor_reset",
    "copy_previous_reset", "running_max_reset",
    "clrs_minimum", "clrs_kadane",
]


def main():
    if len(sys.argv) < 4:
        print(f"Usage: python -m tasks.run_task <task_name> <model_id> <mode> [trials_per_cell]")
        print(f"task_name options: {TASK_NAMES}")
        sys.exit(1)

    task_name, model_id, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    trials_per_cell = int(sys.argv[4]) if len(sys.argv) > 4 else None

    if task_name not in TASK_NAMES:
        print(f"Unknown task '{task_name}'. Options: {TASK_NAMES}")
        sys.exit(1)

    engine = importlib.import_module(f"tasks.{task_name}.engine")
    model_tag = model_id.split("/")[-1]
    output_csv = os.path.join(REPO_ROOT, "results_clean", task_name, mode, f"{model_tag}.csv")

    runner.run(engine, model_id, mode, output_csv, trials_per_cell=trials_per_cell)


if __name__ == "__main__":
    main()
