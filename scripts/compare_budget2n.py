#!/usr/bin/env python3
"""Old budget vs 2N+20: does the generation budget move any reported number?

Pairs each stored grid (results_clean/<task>/grid/<model>.csv, budget int(1.5N)+20 at k != 1)
with its rerun under 2N+20 at every spacing (results_clean/<task>/grid_budget2n/<model>.csv,
written by tasks/rerun_budget2n.py). Everything else about the two runs is identical: same
seeds, trials, lengths, spacings, alphabet values, greedy decoding.

Reports, per (task, model):
  base_64        no-marker token accuracy (TAR) at N = 64
  bounds_64      marker accuracy at N = 64, k in {1, 2, 4, 8, 16} (k = 3 excluded as in Fig. 1)
  delta_64       bounds_64 - base_64
  pooled_*       the paper's length-matched pooled estimator over N <= 64 (tasks/common/pooled)
  k1_identical   share of k = 1 rows whose answer is unchanged. The k = 1 budget was already
                 2N+20, so these rows must reproduce exactly; anything below 1.0 means the
                 rerun is not comparable and the rest of the table should not be trusted.
  cut_*          share of N = 64 rows still ending at the budget with a short answer

Usage: python scripts/compare_budget2n.py [--tasks t1,t2] [--models m1,m2]
Output: results_clean/_summary/budget2n_comparison.csv (and a printed table)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.pooled import pooled_delta  # noqa: E402

TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}
KS = [1, 2, 4, 8, 16]


def read(path):
    if not path.exists():
        return None
    d = pd.read_csv(path, dtype=TEXT, keep_default_na=False)
    d["Interval"] = d.Interval.astype(int)
    d["Seq_Length"] = d.Seq_Length.astype(int)
    d["Trial"] = d.Trial.astype(int)
    d["TAR"] = d.TAR.astype(float)
    return d


def at64(d):
    x = d[d.Seq_Length == 64]
    base = x[x.Method == "Base"].TAR.mean()
    bnd = x[(x.Method == "Cellular_Bounds") & x.Interval.isin(KS)].TAR.mean()
    return base, bnd, bnd - base


def key_cols(d):
    axis = next((c for c in ("Alphabet_Size", "Modulus") if c in d.columns), None)
    return ["Method", "Interval", "Seq_Length", "Trial"] + ([axis] if axis else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="")
    ap.add_argument("--models", default="")
    a = ap.parse_args()

    rows, perk = [], []
    tasks = a.tasks.split(",") if a.tasks else sorted(p.name for p in (REPO / "results_clean").iterdir()
                                                      if (p / "grid_budget2n").is_dir())
    for task in tasks:
        newdir = REPO / "results_clean" / task / "grid_budget2n"
        if not newdir.is_dir():
            continue
        for np_ in sorted(newdir.glob("*.csv")):
            model = np_.stem
            if a.models and model not in a.models.split(","):
                continue
            new, old = read(np_), read(REPO / "results_clean" / task / "grid" / f"{model}.csv")
            if old is None or new is None:
                continue
            kc = key_cols(old)
            m = old.merge(new, on=kc, suffixes=("_old", "_new"))
            k1 = m[m.Interval == 1]
            ob, obd, od = at64(old)
            nb, nbd, nd = at64(new)
            po, pn = pooled_delta(old), pooled_delta(new)
            x_old, x_new = old[old.Seq_Length == 64], new[new.Seq_Length == 64]
            rows.append(dict(
                task=task, model=model, n_paired=len(m), n_old=len(old), n_new=len(new),
                k1_identical=round((k1.LLM_Output_old == k1.LLM_Output_new).mean(), 4) if len(k1) else float("nan"),
                base64_old=round(ob, 4), base64_new=round(nb, 4), base64_diff=round(nb - ob, 4),
                bounds64_old=round(obd, 4), bounds64_new=round(nbd, 4),
                delta64_old=round(od, 4), delta64_new=round(nd, 4), delta64_diff=round(nd - od, 4),
                pooled_base_old=round(po["base"], 4) if po else None,
                pooled_base_new=round(pn["base"], 4) if pn else None,
                pooled_delta_old=round(po["delta"], 4) if po else None,
                pooled_delta_new=round(pn["delta"], 4) if pn else None,
                short64_old=round((x_old.LLM_Output.str.len() < x_old.Ground_Truth.str.len()).mean(), 4),
                short64_new=round((x_new.LLM_Output.str.len() < x_new.Ground_Truth.str.len()).mean(), 4),
            ))
            for k, g in x_old.groupby("Interval"):
                h = x_new[x_new.Interval == k]
                perk.append(dict(task=task, model=model, k=k,
                                 tar_old=round(g.TAR.mean(), 4), tar_new=round(h.TAR.mean(), 4),
                                 short_old=round((g.LLM_Output.str.len() < g.Ground_Truth.str.len()).mean(), 4),
                                 short_new=round((h.LLM_Output.str.len() < h.Ground_Truth.str.len()).mean(), 4)))
    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit("no paired grids found")
    dest = REPO / "results_clean" / "_summary" / "budget2n_comparison.csv"
    out.to_csv(dest, index=False)
    dest_k = REPO / "results_clean" / "_summary" / "budget2n_comparison_by_k.csv"
    pd.DataFrame(perk).to_csv(dest_k, index=False)
    pd.set_option("display.width", 250)
    print(out[["task", "model", "n_paired", "k1_identical", "base64_old", "base64_new",
               "delta64_old", "delta64_new", "delta64_diff", "short64_old", "short64_new"]].to_string(index=False))
    print(f"\nwrote {dest}\nwrote {dest_k}")


if __name__ == "__main__":
    main()
