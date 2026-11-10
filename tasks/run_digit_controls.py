"""Run one tokenization-control task for one model, at N in {4, 8, 16, 32, 64}.

Digit tasks (*_digits) are exact relabelings of the letter position-local tasks; spaced
tasks (*_spaced) are the same letter tasks with characters separated by spaces (one token
per letter); see their engine docstrings and tasks/common/spaced.py.
Lengths stop at 64, the range pooled in Figure 1 and in the per-position analysis.

Usage:
    python -m tasks.run_digit_controls <task> <hf_model_id> [trials_per_cell=20]
Output: results_clean/<task>/grid/<model_tag>.csv
"""

import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.common import runner  # noqa: E402
from tasks.common.boundary import STANDARD_LENGTHS  # noqa: E402

DIGIT_TASKS = ("cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits",
               "cyclic_shift_spaced", "random_permutation_spaced", "conditional_shift_spaced",
               "running_xor_neutral", "running_xor_across", "running_sum_local_neutral", "running_sum_local_across")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in DIGIT_TASKS:
        sys.exit(f"Usage: python -m tasks.run_digit_controls <{'|'.join(DIGIT_TASKS)}> <hf_model_id> [trials]")
    task, model_id = sys.argv[1], sys.argv[2]
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    engine = importlib.import_module(f"tasks.{task}.engine")
    engine.LENGTHS = list(STANDARD_LENGTHS)
    out = os.path.join(REPO_ROOT, "results_clean", task, "grid", f"{model_id.split('/')[-1]}.csv")
    if os.path.exists(out):
        sys.exit(f"Refusing to overwrite {out}; move it aside first.")
    runner.run(engine, model_id, "grid", out, trials_per_cell=trials)


if __name__ == "__main__":
    main()
