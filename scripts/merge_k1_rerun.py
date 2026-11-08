"""Patch the re-run k=1 rows (tasks/rerun_capped_k1.py) into the source result CSVs.

For each staging file results_clean/_k1_rerun/<task>/<name>, the source is
results_clean/<task>/grid/<name>. Every file is read and written as text, so rows that
are not replaced keep their exact original text (checked: a read/write round trip must
reproduce the file byte for byte, or the file is skipped).

Checks per file before patching:
  - every re-run input equals the stored Input_Given;
  - the staging file covers every capped row of the source (else skipped unless --partial).
Reported per file: rows, prefix match (new answer starts with the old cut-off answer),
mean TAR old -> new, and answers still shorter than the truth.

Columns replaced: those the staging row shares with the source (Ground_Truth, LLM_Output,
TAR, SAR, First_Error_Pos, Raw_Model_Output where present); Degenerate is recomputed.

Usage (repo root):
    .conda/envs/qwen-focus/bin/python scripts/merge_k1_rerun.py --dry-run
    .conda/envs/qwen-focus/bin/python scripts/merge_k1_rerun.py [--partial]
"""

import glob
import importlib
import io
import os
import sys
from collections import Counter

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tasks.rerun_capped_k1 import STAGING, is_capped  # noqa: E402

KEEP = ("Ground_Truth", "LLM_Output", "TAR", "SAR", "First_Error_Pos", "Raw_Model_Output")


def is_degenerate(s):  # same rule as tasks/run_task_second_batch.py
    s = str(s)
    if len(s) < 5:
        return False
    return Counter(s).most_common(1)[0][1] / len(s) > 0.85


def read_text(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def to_text(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def keys(df, axis):
    cols = ["Seq_Length", "Trial"] + ([axis] if axis else [])
    return [tuple(int(float(v)) for v in row) for row in df[cols].itertuples(index=False)]


def main():
    dry = "--dry-run" in sys.argv
    partial = "--partial" in sys.argv
    rows_out = []
    for stg_path in sorted(glob.glob(os.path.join(STAGING, "*", "*.csv"))):
        task, name = stg_path.split(os.sep)[-2], os.path.basename(stg_path)
        src_path = os.path.join(REPO, "results_clean", task, "grid", name)
        engine = importlib.import_module(f"tasks.{task}.engine")
        axis = engine.AXIS_NAME

        with open(src_path, newline="") as f:
            original = f.read()
        src = read_text(src_path)
        if to_text(src) != original:
            rows_out.append(dict(task=task, file=name, status="SKIP: round trip not byte-identical"))
            continue
        stg = read_text(stg_path)

        capped = (src.Method == "Cellular_Bounds") & (src.Interval.astype(float) == 1)
        capped &= src.Seq_Length.astype(float).astype(int).map(lambda N: is_capped(engine, N, 1))
        src_idx = {k: i for k, i in zip(keys(src[capped], axis), src[capped].index)}
        stg_keys = keys(stg, axis)
        missing = set(src_idx) - set(stg_keys)
        extra = set(stg_keys) - set(src_idx)
        if extra:
            rows_out.append(dict(task=task, file=name, status=f"SKIP: {len(extra)} staging rows not capped in source"))
            continue
        if missing and not partial:
            rows_out.append(dict(task=task, file=name, status=f"SKIP: incomplete ({len(stg_keys)}/{len(src_idx)})"))
            continue

        n_in_mismatch = n_prefix = n_short = 0
        old_tar, new_tar = [], []
        patched = src.copy()
        for k, (_, s) in zip(stg_keys, stg.iterrows()):
            i = src_idx[k]
            if s.Input_Given != src.at[i, "Input_Given"]:
                n_in_mismatch += 1
                continue
            old_pred = src.at[i, "LLM_Output"]
            n_prefix += s.LLM_Output[:len(old_pred)] == old_pred
            n_short += len(s.LLM_Output) < len(s.Ground_Truth)
            old_tar.append(float(src.at[i, "TAR"]))
            new_tar.append(float(s.TAR))
            for c in KEEP:
                if c in patched.columns and c in stg.columns:
                    patched.at[i, c] = s[c]
            if "Degenerate" in patched.columns:
                patched.at[i, "Degenerate"] = str(is_degenerate(s.LLM_Output))
        n = len(old_tar)
        status = "SKIP: input mismatch" if n_in_mismatch else ("dry-run" if dry else "patched")
        if not dry and not n_in_mismatch:
            with open(src_path, "w", newline="") as f:
                f.write(to_text(patched))
        rows_out.append(dict(
            task=task, file=name, status=status, rows=f"{n}/{len(src_idx)}",
            prefix_match=round(n_prefix / max(n, 1), 3), still_short=round(n_short / max(n, 1), 3),
            tar_old=round(sum(old_tar) / max(n, 1), 3), tar_new=round(sum(new_tar) / max(n, 1), 3),
        ))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(pd.DataFrame(rows_out).to_string(index=False))


if __name__ == "__main__":
    main()
