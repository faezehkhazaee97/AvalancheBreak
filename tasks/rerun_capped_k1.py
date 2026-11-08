"""Re-run the grid cells whose generation budget was too small to hold a correct answer.

With a marker after every character (k = 1) the correct marked output is 2N-1 tokens for
tokenizers that split letters and markers, which exceeded the old budget int(1.5N)+20 once
N > 42: every k = 1 output at N >= 64 was cut off (58 of 64 characters at N = 64), for every
local model. runner.max_new_tokens now gives k = 1 a budget of 2N+20. Decoding is greedy,
so the larger budget only extends outputs that were cut off; no other cell needs rerunning.

For each source CSV of the given model, this regenerates exactly the affected rows
(Cellular_Bounds, k = 1, N where 2N-1 > int(1.5N)+20) and appends them, one at a time, to
a staging CSV (resumable; rows already there are skipped):
    results_clean/_k1_rerun/<task>/<source file name>
Each staging row records Input_Match (rebuilt input == stored Input_Given) and
Prefix_Match (the new answer starts with the old, cut-off answer: same prompt, same
decoding). scripts/merge_k1_rerun.py validates the staging files and patches the sources.

Usage:
    python -m tasks.rerun_capped_k1 <hf_model_id>            # GPU job
    python -m tasks.rerun_capped_k1 <hf_model_id> --dry-run  # no model: count rows and
                                                             # check every input rebuilds
"""

import csv
import glob
import importlib
import os
import random
import sys

import pandas as pd

from tasks.common.runner import _local_seed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = [
    # core grid (Figure 1)
    "cyclic_shift", "random_permutation", "conditional_shift",
    "copy_previous", "running_xor", "running_sum_local", "sliding_window_max",
    "copy_previous_reset", "running_xor_reset", "running_sum_local_reset", "sliding_window_max_reset",
    # external tasks reported in the paper
    "procbench_substitute", "string_copy",
]
STAGING = os.path.join(REPO, "results_clean", "_k1_rerun")
EXTRA_COLS = ["Old_TAR", "Old_Pred_Len", "Input_Match", "Prefix_Match", "Max_New_Tokens"]
# Read answers as text: all-digit answers ("0110...") would otherwise parse as numbers
# and lose leading zeros.
TEXT_COLS = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}


def read_source(path):
    return pd.read_csv(path, dtype=TEXT_COLS, keep_default_na=False)


def old_budget(N):
    return int(N * 1.5) + 20


def is_capped(engine, N, k):
    """True where a correct k=1 answer could not fit in the old budget."""
    if hasattr(engine, "estimate_max_new_tokens") or not engine.OUTPUT_HAS_BOUNDARIES:
        return False
    return k == 1 and 2 * N - 1 > old_budget(N)


def source_files(task, tag):
    out = []
    for f in sorted(glob.glob(os.path.join(REPO, "results_clean", task, "grid", "*.csv"))):
        name = os.path.basename(f)
        if "POOLED" in name:
            continue
        if name == f"{tag}.csv" or name.startswith(f"{tag}_"):
            out.append(f)
    return out


def selected_rows(df, engine):
    m = (df.Method == "Cellular_Bounds") & (df.Interval == 1)
    m &= df.Seq_Length.map(lambda N: is_capped(engine, int(N), 1))
    return df[m]


def key_of(row, axis_name):
    key = (int(row["Seq_Length"]), int(row["Trial"]))
    return key + ((int(row[axis_name]),) if axis_name else ())


def done_keys(path, axis_name):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return {key_of(r, axis_name) for _, r in d.iterrows()}


def append_row(path, row, fieldnames):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    model_id = sys.argv[1]
    dry = "--dry-run" in sys.argv
    tag = model_id.split("/")[-1]

    plan, mismatches = [], 0
    for task in TASKS:
        engine = importlib.import_module(f"tasks.{task}.engine")
        axis_name = engine.AXIS_NAME
        for src in source_files(task, tag):
            df = read_source(src)
            sel = selected_rows(df, engine)
            if sel.empty:
                continue
            for _, r in sel.iterrows():
                N, t = int(r.Seq_Length), int(r.Trial)
                axis = int(r[axis_name]) if axis_name else None
                rng = random.Random(_local_seed(N, 1, axis, t))
                if axis_name:
                    input_str, _, _ = engine.build_prompt(N, 1, rng, axis_val=axis)
                else:
                    input_str, _, _ = engine.build_prompt(N, 1, rng)
                if input_str != str(r.Input_Given):
                    mismatches += 1
            plan.append((task, engine, src, sel))
            print(f"{task:26s} {os.path.basename(src):42s} rows={len(sel):4d} N={sorted(sel.Seq_Length.unique())}")

    total = sum(len(s) for *_, s in plan)
    print(f"\n{tag}: {total} rows in {len(plan)} files; input mismatches: {mismatches}")
    if dry:
        return
    if mismatches:
        sys.exit(f"Refusing to run: {mismatches} inputs do not rebuild identically.")

    import torch  # noqa: F401  (imported here so --dry-run needs no GPU stack)
    from transformers import AutoTokenizer
    from tasks.common.runner import _run_one, _seed_everything, load_model, max_new_tokens

    _seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id)
    model.eval()

    n_done = n_prefix = 0
    for task, engine, src, sel in plan:
        axis_name = engine.AXIS_NAME
        out = os.path.join(STAGING, task, os.path.basename(src))
        skip = done_keys(out, axis_name)
        fieldnames = None
        for _, r in sel.iterrows():
            if key_of(r, axis_name) in skip:
                continue
            N, t = int(r.Seq_Length), int(r.Trial)
            axis = int(r[axis_name]) if axis_name else None
            row = _run_one(engine, tokenizer, model, N, 1, axis, t)
            old_pred = str(r.LLM_Output)
            new_pred = "" if row["LLM_Output"] is None else str(row["LLM_Output"])
            row.update(
                Old_TAR=r.TAR, Old_Pred_Len=len(old_pred),
                Input_Match=row["Input_Given"] == str(r.Input_Given),
                Prefix_Match=new_pred[:len(old_pred)] == old_pred,
                Max_New_Tokens=max_new_tokens(engine, N, 1),
            )
            if fieldnames is None:
                fieldnames = list(row.keys())
            append_row(out, row, fieldnames)
            n_done += 1
            n_prefix += row["Prefix_Match"]
            if n_done == 10 and n_prefix == 0:
                sys.exit("First 10 reruns all diverge from the original outputs: prompt or "
                         "decoding differs from the original run. Stopping.")
            if n_done % 25 == 0:
                print(f"  {n_done}/{total} rows; prefix match so far {n_prefix}/{n_done}", flush=True)
        print(f"done: {task} {os.path.basename(src)}", flush=True)
    print(f"Finished {n_done} new rows; prefix match {n_prefix}/{max(n_done, 1)}")


if __name__ == "__main__":
    main()
