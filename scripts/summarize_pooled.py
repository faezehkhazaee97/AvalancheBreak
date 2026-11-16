#!/usr/bin/env python3
"""Every pooled marker effect in the paper, with the length-matched estimator
(tasks/common/pooled.py), written to results_clean/_summary/pooled_all.csv.

family: letters | spaced | digits | dependency | reset | gemini_table | external
Main-grid families pool N <= 64; the Gemini table and external tasks pool every tested length.

`z` is the primary, stratified z of tasks/common/pooled.py (the (N, k, A) cells of the grid are
fixed design strata); `z_unstrat` is the older form that pools all of a condition's generations
into one s/sqrt(n). The script prints the |z| >= 3.54 counts the paper quotes, under both.
"""

import glob
import pathlib
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.pooled import pooled_delta  # noqa: E402

MODELS = {"Qwen2.5 7B": "Qwen2.5-7B-Instruct", "Qwen2.5 14B": "Qwen2.5-14B-Instruct",
          "Qwen2.5 32B": "Qwen2.5-32B-Instruct", "Mistral 7B": "Mistral-7B-Instruct-v0.3",
          "Phi-3.5 mini": "Phi-3.5-mini-instruct", "Qwen3.8 27B": "Qwen3.8-27B", "Gemma 4 31B": "gemma-4-31B-it"}
PL = ["cyclic_shift", "random_permutation", "conditional_shift"]
DEP = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]
L64 = [4, 8, 16, 32, 64]
SIG_Z = 3.54   # Bonferroni |z| across the 125 pooled cells of the main analysis (Section 3.4)
GEMINI = REPO / "results_clean" / "gemini_pilot" / "gemini-flash-latest_POOLED_3trials.csv"
GEMINI36 = REPO / "results_clean" / "gemini_pilot" / "gemini-3.6-flash_POOLED.csv"


def grid(task, tag, n128=False):
    # The tag must match the whole stem or be followed by "_", as in analyze_per_position.load:
    # a bare "{tag}*" glob also swept up the LoRA runs ("{tag}-lorabase.csv", "-lorasep.csv",
    # "-lorabase-matched.csv"), pooling adapter generations into the base model's effect on the
    # four dependency tasks and on random_permutation_digits at Qwen2.5-7B and 32B.
    fs = [f for f in sorted(glob.glob(str(REPO / "results_clean" / task / "grid" / f"{tag}*.csv")))
          if "POOLED" not in f and (n128 or "N128" not in f)
          and (pathlib.Path(f).name == f"{tag}.csv" or pathlib.Path(f).name.startswith(f"{tag}_"))]
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    axis = next((c for c in ("Alphabet_Size", "Modulus") if c in d.columns), None)
    return d.drop_duplicates(["Method", "Interval", "Seq_Length", "Trial"] + ([axis] if axis else []))


def main():
    rows = []

    def add(family, task, model, d, lengths):
        if d is None:
            return
        r = pooled_delta(d, lengths=lengths)
        if r:
            rows.append(dict(family=family, task=task, model=model, **r))

    gem = pd.read_csv(GEMINI)
    for model, tag in MODELS.items():
        for t in PL:
            add("letters", t, model, grid(t, tag), L64)
            add("spaced", t, model, grid(t + "_spaced", tag), L64)
            add("digits", t, model, grid(t + "_digits", tag), L64)
        for t in DEP:
            add("dependency", t, model, grid(t, tag), L64)
            add("reset", t, model, grid(t + "_reset", tag), L64)
    for t in PL:
        add("letters", t, "Gemini 3.5 Flash", gem[gem.Task == t], L64)
    for t in DEP:
        add("dependency", t, "Gemini 3.5 Flash", gem[gem.Task == t], L64)
    for t in sorted(gem.Task.unique()):
        add("gemini_table", t, "Gemini 3.5 Flash", gem[gem.Task == t], None)
    if GEMINI36.exists():
        g6 = pd.read_csv(GEMINI36)
        for t in PL:
            add("letters", t, "Gemini 3.6 Flash", g6[g6.Task == t], L64)
            add("spaced", t, "Gemini 3.6 Flash", g6[g6.Task == t + "_spaced"], L64)
            add("digits", t, "Gemini 3.6 Flash", g6[g6.Task == t + "_digits"], L64)
        for t in DEP:
            add("dependency", t, "Gemini 3.6 Flash", g6[g6.Task == t], L64)
        for t in sorted(g6.Task.unique()):
            add("gemini_table", t, "Gemini 3.6 Flash", g6[g6.Task == t], None)
    for model in ["Qwen2.5 7B", "Qwen2.5 32B"]:
        tag = MODELS[model]
        add("external", "procbench_substitute", model, grid("procbench_substitute", tag), None)
        add("external", "string_copy", model, grid("string_copy", tag), None)
        add("external", "clrs_minimum", model, grid("clrs_minimum", tag), None)
        add("external", "clrs_kadane", model, grid("clrs_kadane", tag), None)
        v = REPO / "results" / tag / "vowel_tagging_boundary_sweep.csv"
        if v.exists():
            d = pd.read_csv(v)
            d["Method"] = (d.Interval > 0).map({True: "Cellular_Bounds", False: "Base"})
            add("external", "vowel_tagging", model, d, None)
    gc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(REPO / "results_clean" / "gemini_pilot" /
                                                                    "gemini-3.5-flash_full_grid_trial*.csv")))])
    for t in ["clrs_minimum", "clrs_kadane"]:
        add("external", t, "Gemini 3.5 Flash", gc[gc.Task == t], None)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(REPO / "results_clean" / "_summary" / "pooled_all.csv", index=False)
    print(f"Wrote {len(out)} pooled cells to results_clean/_summary/pooled_all.csv")
    if out.thin_strata.sum():
        print(f"WARNING: {int(out.thin_strata.sum())} strata hold a single generation "
              "(given the pooled within-stratum variance)")
    print("\n|z| >= 3.54, unstratified (old) -> stratified (new), by family:")
    for fam, g in out.groupby("family", sort=False):
        s_new, s_old = g.z.abs() >= SIG_Z, g.z_unstrat.abs() >= SIG_Z
        print(f"  {fam:12s} {len(g):3d} cells: significant {s_old.sum():3d} -> {s_new.sum():3d}; "
              f"positive {int((g.z_unstrat >= SIG_Z).sum()):3d} -> {int((g.z >= SIG_Z).sum()):3d}; "
              f"negative {int((g.z_unstrat <= -SIG_Z).sum()):3d} -> {int((g.z <= -SIG_Z).sum()):3d}; "
              f"all positive delta: {bool((g.delta > 0).all())}")
        for r in g.itertuples():
            if (abs(r.z) >= SIG_Z) != (abs(r.z_unstrat) >= SIG_Z):
                print(f"      FLIPS {'to significant' if abs(r.z) >= SIG_Z else 'to NOT significant'}: "
                      f"{r.task} {r.model}  z {r.z_unstrat:+.2f} -> {r.z:+.2f}  (delta {r.delta:+.3f})")
    print(f"\nstratified |z| larger than unstratified in "
          f"{int((out.z.abs() > out.z_unstrat.abs()).sum())} of {len(out)} cells")


if __name__ == "__main__":
    main()
