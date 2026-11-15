#!/usr/bin/env python3
"""Audit of the k = 1 budget rerun (tasks/rerun_capped_k1.py, merged by scripts/merge_k1_rerun.py).

With greedy decoding a larger budget should only extend an answer that was cut off. The staging
files (results_clean/_k1_rerun/<task>/<file>) record Prefix_Match: whether the new answer starts
with the old, cut-off one. For every staging row that does not, this script recovers the old answer
from git (the source CSVs before the merge commit, read as text) and reports
  - which model x task files diverge, and how often;
  - the shared prefix: the number of leading answer characters the old and new outputs agree on,
    absolutely and as a share of the old (cut-off) answer;
  - accuracy of the old and the new answer over the old answer's span.
Some staging rows are flagged only because the rerun read an all-digit old answer as a number
(losing its leading zeros, so Old_Pred_Len < the true length); read as text, the old answer is a
prefix of the new one. Those rows are counted separately as spurious and not as divergences.
Then it recomputes every main-grid pooled effect (tasks/common/pooled.py, N <= 64, as in
scripts/summarize_pooled.py) for the affected model x task cells (i) with the diverging rows dropped
and (ii) with every regenerated row of the cell dropped, and reports the largest change.

Output: results_clean/_summary/k1_rerun_divergence.csv, results_clean/_summary/k1_rerun_pooled_check.csv
"""

import glob
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from tasks.common.pooled import pooled_delta  # noqa: E402
from summarize_pooled import MODELS  # noqa: E402

MERGE_COMMIT = "d3b5383"   # "Merge the k=1 budget rerun": its parent holds the original rows
STAGING = REPO / "results_clean" / "_k1_rerun"
L64 = [4, 8, 16, 32, 64]
TAG2LAB = {v: k for k, v in MODELS.items()}
TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}


def old_source(task, name):
    rel = f"results_clean/{task}/grid/{name}"
    txt = subprocess.run(["git", "-C", str(REPO), "show", f"{MERGE_COMMIT}^:{rel}"], capture_output=True, text=True)
    if txt.returncode:
        return None
    return pd.read_csv(io.StringIO(txt.stdout), dtype=TEXT, keep_default_na=False)


def axis_of(d):
    return next((c for c in ("Alphabet_Size", "Modulus") if c in d.columns), None)


def key_cols(d):
    a = axis_of(d)
    return ["Seq_Length", "Trial"] + ([a] if a else [])


def tag_of(name):
    stem = name[:-4]
    for tag in sorted(TAG2LAB, key=len, reverse=True):
        if stem == tag or stem.startswith(tag + "_"):
            return tag
    return None


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def acc(truth, pred, n):
    return np.mean([i < len(pred) and pred[i] == truth[i] for i in range(min(n, len(truth)))]) if n else np.nan


def main():
    div, regen = [], []
    for f in sorted(glob.glob(str(STAGING / "*" / "*.csv"))):
        task, name = f.split(os.sep)[-2], os.path.basename(f)
        tag = tag_of(name)
        stg = pd.read_csv(f, dtype=TEXT, keep_default_na=False)
        kc = key_cols(stg)
        for _, r in stg.iterrows():
            regen.append(dict(task=task, tag=tag, file=name, **{c: int(float(r[c])) for c in kc},
                              diverged=str(r.Prefix_Match) != "True"))
        bad = stg[stg.Prefix_Match.astype(str) != "True"]
        if bad.empty:
            continue
        old = old_source(task, name)
        if old is None:
            print("no git copy of", task, name)
            continue
        old = old[(old.Method == "Cellular_Bounds") & (old.Interval.astype(float) == 1)]
        old_by = {tuple(int(float(v)) for v in row[kc]): row for _, row in old.iterrows()}
        for _, r in bad.iterrows():
            o = old_by.get(tuple(int(float(r[c])) for c in kc))
            if o is None:
                continue
            L = common_prefix(o.LLM_Output, r.LLM_Output)
            n_old = len(o.LLM_Output)
            div.append(dict(task=task, model=TAG2LAB.get(tag, tag), file=name, N=int(float(r.Seq_Length)),
                            trial=int(r.Trial), old_len=n_old, shared_prefix=L, prefix_share=L / max(n_old, 1),
                            old_acc_over_old_span=acc(r.Ground_Truth, o.LLM_Output, n_old),
                            new_acc_over_old_span=acc(r.Ground_Truth, r.LLM_Output, n_old),
                            old_tar=float(o.TAR), new_tar=float(r.TAR)))
    dv = pd.DataFrame(div)
    dv["genuine"] = dv.shared_prefix < dv.old_len
    print("Flagged rows whose old answer, read as text, IS a prefix of the new one (leading-zero artifact):")
    print(dv[~dv.genuine].groupby(["model", "task"]).size().to_string())
    spurious = set(zip(dv[~dv.genuine].task, dv[~dv.genuine].file, dv[~dv.genuine].N, dv[~dv.genuine].trial))
    dv = dv[dv.genuine]
    rg = pd.DataFrame(regen)
    rg["diverged"] &= ~pd.Series([(t, f, n, tr) in spurious for t, f, n, tr in zip(rg.task, rg.file, rg.Seq_Length, rg.Trial)],
                                 index=rg.index)
    out = REPO / "results_clean" / "_summary"
    dv.round(4).to_csv(out / "k1_rerun_divergence.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    n_all = rg.groupby(["task", "tag"]).size().rename("regenerated")
    summ = dv.groupby(["model", "task"]).agg(diverged=("trial", "size"), N=("N", lambda s: ",".join(map(str, sorted(set(s))))),
                                             prefix_median=("shared_prefix", "median"), prefix_min=("shared_prefix", "min"),
                                             prefix_max=("shared_prefix", "max"), old_len=("old_len", "median"),
                                             share_median=("prefix_share", "median"),
                                             old_acc=("old_acc_over_old_span", "mean"), new_acc=("new_acc_over_old_span", "mean"),
                                             old_tar=("old_tar", "mean"), new_tar=("new_tar", "mean"))
    print("\nRegenerated k = 1 rows whose new answer does not extend the old one:")
    print(summ.round(3).to_string())
    print(f"\nAll diverging rows: {len(dv)} of {len(rg)} regenerated; shared prefix median {dv.shared_prefix.median():.0f} "
          f"characters (IQR {dv.shared_prefix.quantile(.25):.0f}-{dv.shared_prefix.quantile(.75):.0f}), "
          f"median share of the old answer {dv.prefix_share.median():.2f}")
    for N, g in dv.groupby("N"):
        print(f"  N = {N}: {len(g)} rows, shared prefix median {g.shared_prefix.median():.0f} of old length "
              f"{g.old_len.median():.0f}; accuracy over the old span old {g.old_acc_over_old_span.mean():.3f}, "
              f"new {g.new_acc_over_old_span.mean():.3f}")

    # pooled effects without the regenerated rows
    rows = []
    cells = rg[rg.diverged].groupby(["task", "tag"]).size().index
    for task, tag in cells:
        lab = TAG2LAB.get(tag)
        if lab is None:
            continue
        fam_task = [("dependency" if task in ("copy_previous", "running_xor", "running_sum_local", "sliding_window_max")
                     else "reset" if task.endswith("_reset") else "letters", task)]
        fs = [f for f in sorted(glob.glob(str(REPO / "results_clean" / task / "grid" / f"{tag}*.csv")))
              if "POOLED" not in f and "N128" not in f and tag_of(os.path.basename(f)) == tag]
        if not fs:
            continue
        d = pd.concat([pd.read_csv(f).assign(_file=os.path.basename(f)) for f in fs], ignore_index=True)
        kc = key_cols(d)
        d = d.drop_duplicates(["Method", "Interval"] + kc)
        r_cell = rg[(rg.task == task) & (rg.tag == tag)]
        is_k1 = (d.Method == "Cellular_Bounds") & (d.Interval.astype(int) == 1)

        def mask(sel):
            keys = {(fn,) + tuple(int(x[c]) for c in kc) for fn, x in zip(sel.file, sel.to_dict("records"))}
            return is_k1 & pd.Series([(fn,) + tuple(int(float(v)) for v in row) for fn, row in
                                      zip(d._file, d[kc].itertuples(index=False))], index=d.index).isin(keys)

        full = pooled_delta(d, lengths=L64)
        no_div = pooled_delta(d[~mask(r_cell[r_cell.diverged])], lengths=L64)
        no_regen = pooled_delta(d[~mask(r_cell)], lengths=L64)
        for fam, t in fam_task:
            rows.append(dict(family=fam, task=t, model=lab, delta=full["delta"],
                             delta_without_diverged=no_div["delta"], delta_without_regenerated=no_regen["delta"],
                             n_bounds=full["n_bounds"], n_diverged_dropped=full["n_bounds"] - no_div["n_bounds"],
                             n_regenerated_dropped=full["n_bounds"] - no_regen["n_bounds"]))
    pc = pd.DataFrame(rows)
    pc["change_div"] = pc.delta_without_diverged - pc.delta
    pc["change_regen"] = pc.delta_without_regenerated - pc.delta
    pc.round(4).to_csv(out / "k1_rerun_pooled_check.csv", index=False)
    print("\nPooled effect (N <= 64, length-matched) with the regenerated k = 1 rows dropped:")
    print(pc.round(4).to_string(index=False))
    print(f"largest |change|: without diverging rows {pc.change_div.abs().max():.4f}; "
          f"without every regenerated row of the cell {pc.change_regen.abs().max():.4f}")


if __name__ == "__main__":
    main()
