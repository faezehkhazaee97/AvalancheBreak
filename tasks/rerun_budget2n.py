"""Re-run one task's full grid for one model under a larger generation budget.

Background. The boundary-task budget is int(1.5N)+20 new tokens for the no-marker run and
every spacing k >= 2, and 2N+20 at k = 1 (tasks/common/runner.max_new_tokens). In a spaced
format a correct answer of length N is written with single spaces between characters. For a
tokenizer that merges " X" into one token that still costs about N tokens, but for one that
emits the space separately -- which is what Qwen, Gemma and Phi do for *digit* characters --
it costs 2N-1, which at N = 64 is 127 against a budget of 116. Those cells were truncated.

This script reruns a whole grid with the budget override AB_GEN_BUDGET (default 2n, i.e.
2N+20 at every k) and writes it to

    results_clean/<task>/grid_budget2n/<model_tag>.csv

a sibling of the stored grid, never inside it. Every analysis loader in this repo globs
inside results_clean/<task>/grid/ and none of them recurse, so these files are invisible to
the existing analyses; compare them with scripts/compare_budget2n.py instead.

Everything else is identical to the original run: same GLOBAL_SEED, same per-cell local
seeds (N, k, axis_val, trial), same lengths, spacings and alphabet values, same greedy
decoding. Decoding is greedy, so a larger budget can only extend an output that was cut off.

Usage:
    python -m tasks.rerun_budget2n <task> <hf_model_id> [trials_per_cell=20]
    AB_GEN_BUDGET=2.5n python -m tasks.rerun_budget2n <task> <hf_model_id>
"""

import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.common import runner  # noqa: E402
from tasks.common.boundary import STANDARD_LENGTHS  # noqa: E402

SUBDIR = "grid_budget2n"


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: python -m tasks.rerun_budget2n <task> <hf_model_id> [trials_per_cell=20]")
    task, model_id = sys.argv[1], sys.argv[2]
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    os.environ.setdefault("AB_GEN_BUDGET", "2n")
    engine = importlib.import_module(f"tasks.{task}.engine")
    # the position-local controls are run at N <= 64 by tasks/run_digit_controls.py; the
    # real-sequence tasks carry their own LENGTHS and keep them
    if not hasattr(engine, "LENGTHS"):
        engine.LENGTHS = list(STANDARD_LENGTHS)

    out = os.path.join(REPO_ROOT, "results_clean", task, SUBDIR, f"{model_id.split('/')[-1]}.csv")
    if os.path.exists(out):
        sys.exit(f"Refusing to overwrite {out}; move it aside first.")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"budget override AB_GEN_BUDGET={os.environ['AB_GEN_BUDGET']} "
          f"-> max_new_tokens(N=64, k=2) = {runner.max_new_tokens(engine, 64, 2)}")
    runner.run(engine, model_id, "grid", out, trials_per_cell=trials)


if __name__ == "__main__":
    main()
