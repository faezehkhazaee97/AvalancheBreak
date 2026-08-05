#!/usr/bin/env python3
"""Q5: reset-instruction compliance audit (CPU-only, uses already-stored
grid data -- no GPU needed). Direct precedent: Appendix B's echo-vs-
segmentation isolation found the model disobeys a "do not echo" instruction
in 90-100% of trials. This asks the analogous question for the reset-variant
tasks: at the position immediately after each boundary (the one position
whose expected behavior actually changes between the original and reset
tasks), does the model's output actually follow RESET semantics, or does it
silently follow the ORIGINAL (carry-over) semantics instead -- i.e., does
it actually comply with "treat this position like position 0" or does it
just keep computing the old recurrence across the boundary as if the reset
instruction weren't there?

If compliance is low, the entire reset-ablation causal story (Section 3.5)
is at risk: the reset variant's changed accuracy might not reflect "the
model successfully severed the dependency" but something else (e.g. it's
just doing the original task and getting scored against reset ground
truth, which would look like "harm" for unrelated reasons).

Method: for each reset task, recompute what the ORIGINAL (non-reset,
continuous) rule would produce at the same "boundary-adjacent" position,
using the identical underlying digit sequence (reconstructed from
Input_Given). Compare the model's actual output character at that position
against (a) the RESET expectation (already stored as Ground_Truth) and
(b) the ORIGINAL/carry-over expectation (recomputed here). Only positions
where (a) != (b) are informative (a genuine fork in what the model could
have done); positions where they coincide can't distinguish compliance
from coincidence and are excluded from the compliance rate.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.running_xor.engine import _compute_xor_chain
from tasks.copy_previous.engine import _compute_shift
from tasks.running_sum_local.engine import _compute_running_sum_local
from tasks.reverse_xor.engine import _compute_xor_chain_reverse
from tasks.running_max.engine import _compute_running_max

RESULTS = REPO_ROOT / "results_clean"
OUT_DIR = RESULTS / "_reset_compliance_audit"


def reset_positions_forward(N, k):
    """Position j is 'immediately after a boundary' (forward-dependency
    tasks: running_xor, copy_previous, running_sum_local, running_max)."""
    return [j for j in range(N) if k > 0 and j % k == 0 and j > 0]


def reset_positions_backward(N, k):
    """Position i is 'immediately before a boundary' (reverse_xor's
    dependency looks forward, so IT is the one that needs resetting)."""
    return [i for i in range(N) if k > 0 and (i + 1) % k == 0 and i != N - 1]


TASKS = {
    "running_xor_reset": {
        "orig_task": "running_xor", "axis_col": None,
        "orig_fn": lambda digits, modulus=None: _compute_xor_chain(digits),
        "positions_fn": reset_positions_forward,
    },
    "copy_previous_reset": {
        "orig_task": "copy_previous", "axis_col": None,
        "orig_fn": lambda digits, modulus=None: _compute_shift(digits),
        "positions_fn": reset_positions_forward,
    },
    "running_sum_local_reset": {
        "orig_task": "running_sum_local", "axis_col": "Modulus",
        "orig_fn": lambda digits, modulus: _compute_running_sum_local(digits, modulus),
        "positions_fn": reset_positions_forward,
    },
    "reverse_xor_reset": {
        "orig_task": "reverse_xor", "axis_col": None,
        "orig_fn": lambda digits, modulus=None: _compute_xor_chain_reverse(digits),
        "positions_fn": reset_positions_backward,
    },
    "running_max_reset": {
        "orig_task": "running_max", "axis_col": "Alphabet_Size",
        "orig_fn": lambda digits, modulus=None: _compute_running_max(digits),
        "positions_fn": reset_positions_forward,
    },
}


def strip_bars(s):
    return str(s).replace("|", "")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    all_rows = []

    for task, spec in TASKS.items():
        f = RESULTS / task / "grid" / "Qwen2.5-7B-Instruct.csv"
        if not f.exists():
            print(f"SKIP {task}: {f} not found")
            continue
        df = pd.read_csv(f)
        df = df[(df.Interval != 3) & (df.Interval > 0)]  # Base has no boundaries to audit

        for _, row in df.iterrows():
            N, k = int(row.Seq_Length), int(row.Interval)
            input_digits = strip_bars(row.Input_Given)
            model_out = str(row.LLM_Output) if pd.notna(row.LLM_Output) else ""
            truth = strip_bars(row.Ground_Truth) if pd.notna(row.Ground_Truth) else str(row.Ground_Truth)

            if len(input_digits) != N:
                continue  # malformed/degenerate row, skip

            axis_val = row[spec["axis_col"]] if spec["axis_col"] else None
            orig_output = spec["orig_fn"](input_digits, axis_val)

            positions = spec["positions_fn"](N, k)
            for p in positions:
                if p >= len(truth) or p >= len(orig_output):
                    continue
                reset_expected = truth[p]
                orig_expected = orig_output[p]
                if reset_expected == orig_expected:
                    continue  # not a distinguishing position
                model_char = model_out[p] if p < len(model_out) else None
                all_rows.append({
                    "Task": task, "Seq_Length": N, "Interval": k, "Trial": row.Trial,
                    "Position": p,
                    "Reset_Expected": reset_expected, "Orig_Expected": orig_expected,
                    "Model_Char": model_char,
                    "Follows_Reset": model_char == reset_expected,
                    "Follows_Orig_CarryOver": model_char == orig_expected,
                    "Follows_Neither": model_char is not None and model_char != reset_expected and model_char != orig_expected,
                })

    audit_df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "reset_compliance_audit_rows.csv"
    audit_df.to_csv(out_path, index=False)
    print(f"Wrote {len(audit_df)} distinguishing-position rows to {out_path}")

    print(f"\n{len(audit_df)} positions where reset and carry-over semantics actually differ "
          f"(non-distinguishing positions excluded).")
    print("\n--- Overall compliance ---")
    print(f"Follows RESET semantics:      {audit_df.Follows_Reset.mean():.3f}")
    print(f"Follows ORIGINAL carry-over:  {audit_df.Follows_Orig_CarryOver.mean():.3f}")
    print(f"Follows neither:              {audit_df.Follows_Neither.mean():.3f}")

    print("\n--- By task ---")
    by_task = audit_df.groupby("Task")[["Follows_Reset", "Follows_Orig_CarryOver", "Follows_Neither"]].mean()
    by_task["N_distinguishing"] = audit_df.groupby("Task").size()
    print(by_task.round(3).to_string())

    print("\n--- By task x length ---")
    by_task_len = audit_df.groupby(["Task", "Seq_Length"])[["Follows_Reset", "Follows_Orig_CarryOver"]].mean()
    by_task_len["N"] = audit_df.groupby(["Task", "Seq_Length"]).size()
    print(by_task_len.round(3).to_string())

    summary_path = OUT_DIR / "reset_compliance_summary.csv"
    by_task.to_csv(summary_path)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
