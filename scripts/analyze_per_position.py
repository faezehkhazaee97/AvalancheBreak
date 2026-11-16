#!/usr/bin/env python3
"""Per-position accuracy: where in the sequence do markers help or hurt?

Positions are classified by the task's lookback horizon h, the number of previous inputs each
output reads: h = 1 for the one-step tasks (y_i reads x_{i-1}), and h = 2 for
sliding_window_max, y_i = max(x_{i-2}, x_{i-1}, x_i), whose first TWO positions after a marker
have an input across it. For a marker spacing k,

  post    : the crossing positions, {i : i >= k, (i mod k) < h}. At h = 1 this is the paper's
            post-marker class, {i > 0 : i mod k = 0}.
  within  : the strictly internal positions, {i > 0 : (i mod k) >= h}, which read nothing
            across a marker. Position 0 is in neither class: dependency tasks copy it as-is, so
            it carries no evidence. The class is empty when k <= h, and such spacings are left
            out of the within-segment comparison (at k = 2 sliding_window_max has no strictly
            internal position at all).
  base    : the no-marker (Base) run, all positions
  base_at_post / base_at_within : Base accuracy at the same position indices as post /
            within (controls for position-in-sequence effects: post positions are not
            uniformly spread when k is large)
Pooled over N in {4, 8, 16, 32, 64}, k in {2, 4, 8, 16} (k = 1 has no within positions;
k = 3 excluded as in Figure 1), and alphabet or modulus values. Accuracy is index-aligned
after stripping markers and spaces, clipped to the truth length (the paper's TAR).
Uncertainty: rows (generations) are the unit; 95% CI = 1.96 x SE of per-row means.

Also writes a distance profile: accuracy by d = i mod k (d = 0 is post-marker) at
k in {8, 16}, for the "decays with distance from the marker" analysis.

Outputs:
  results_clean/_summary/per_position_values.csv
  results_clean/_summary/per_position_distance.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GEMINI = REPO / "results_clean" / "gemini_pilot" / "gemini-flash-latest_POOLED_3trials.csv"
GEMINI36 = REPO / "results_clean" / "gemini_pilot" / "gemini-3.6-flash_POOLED.csv"  # all 13 tasks
LENGTHS = [4, 8, 16, 32, 64]
KS = [2, 4, 8, 16]
BLOCKS = [
    ("Position-local, letters", ["cyclic_shift", "random_permutation", "conditional_shift"]),
    ("Position-local, spaced letters", ["cyclic_shift_spaced", "random_permutation_spaced", "conditional_shift_spaced"]),
    ("Position-local, digits", ["cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"]),
    ("Dependency, as is", ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]),
    ("Dependency severed (reset)", ["copy_previous_reset", "running_xor_reset",
                                    "running_sum_local_reset", "sliding_window_max_reset"]),
]
MODELS = [
    ("Qwen2.5 7B", "Qwen2.5-7B-Instruct"), ("Qwen2.5 14B", "Qwen2.5-14B-Instruct"),
    ("Qwen2.5 32B", "Qwen2.5-32B-Instruct"), ("Mistral 7B", "Mistral-7B-Instruct-v0.3"),
    ("Phi-3.5 mini", "Phi-3.5-mini-instruct"), ("Qwen3.8 27B", "Qwen3.8-27B"),
    ("Gemma 4 31B", "gemma-4-31B-it"), ("Gemini 3.5 Flash", None), ("Gemini 3.6 Flash", "gemini36"),
]
TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}

# Lookback horizon per task: how many previous inputs each output reads. Only the windowed task
# reads more than one, so only it has a second crossing position after each marker.
LOOKBACK = {"sliding_window_max": 2}
# Variant suffixes a task name can carry; the horizon is a property of the rule, not the variant
# (sliding_window_max_reset still rewrites exactly its two crossing positions).
VARIANT_SUFFIXES = ("_reset", "_finek", "_jitter", "_worked", "_spaced", "_digits",
                    "_neutral", "_across", "_natural", "_n128")


def horizon(task):
    """The lookback horizon h of a task or any of its variants (default 1)."""
    t = str(task)
    while True:
        for suf in VARIANT_SUFFIXES:
            if t.endswith(suf):
                t = t[: -len(suf)]
                break
        else:
            return LOOKBACK.get(t, 1)


def position_classes(idx, k, h=1):
    """(crossing, strictly_internal) boolean masks over output indices, for spacing k and
    horizon h.

    crossing = {i : i >= k, (i mod k) < h}: the h positions after each marker, whose inputs
        reach back across it. At h = 1 this is exactly {i > 0 : i mod k = 0}. The bound is
        i >= k rather than i > 0 because the first h positions of the sequence follow no
        marker at all and cross nothing; at h = 1 the two bounds pick out the same set.
    strictly internal = {i > 0 : (i mod k) >= h}: reads nothing across a marker. Empty when
        k <= h, so those spacings carry no within-segment comparison.
    """
    r = idx % k
    return (idx >= k) & (r < h), (idx > 0) & (r >= h)


def load(task, tag):
    if tag == "gemini36":
        if not GEMINI36.exists():
            return None
        d = pd.read_csv(GEMINI36, dtype=TEXT, keep_default_na=False)
        d = d[d.Task == task]
        return d if len(d) else None
    if tag is None:
        if not GEMINI.exists():
            return None
        d = pd.read_csv(GEMINI, dtype=TEXT, keep_default_na=False)
        d = d[d.Task == task]
        return d if len(d) else None
    files = [f for f in sorted((REPO / "results_clean" / task / "grid").glob(f"{tag}*.csv"))
             if (f.name == f"{tag}.csv" or f.name.startswith(f"{tag}_"))
             if "POOLED" not in f.name]
    if not files:
        return None
    d = pd.concat([pd.read_csv(f, dtype=TEXT, keep_default_na=False) for f in files], ignore_index=True)
    axis = next((c for c in ("Alphabet_Size", "Modulus") if c in d.columns), None)
    return d.drop_duplicates(subset=["Method", "Interval", "Seq_Length", "Trial"] + ([axis] if axis else []))


def correct_vector(truth, pred):
    t = truth.replace("|", "").replace(" ", "")
    p = pred.replace("|", "").replace(" ", "")[: len(t)]
    return np.array([i < len(p) and p[i] == ch for i, ch in enumerate(t)], dtype=float)


def summarize(d, h=1):
    d = d[d.Seq_Length.astype(int).isin(LENGTHS)]
    base = d[d.Method == "Base"]
    bnd = d[(d.Method == "Cellular_Bounds") & d.Interval.astype(int).isin(KS)]
    if len(base) < 2 or len(bnd) < 2:
        return None, None
    # Base accuracy by position index, per length (for the index-matched controls)
    base_vecs = {}
    base_rows = []
    for _, r in base.iterrows():
        v = correct_vector(r.Ground_Truth, r.LLM_Output)
        base_vecs.setdefault(len(v), []).append(v)
        base_rows.append(v.mean())
    base_by_idx = {n: np.mean(vs, axis=0) for n, vs in base_vecs.items()}
    post_rows, within_rows, bpost, bwithin = [], [], [], []
    dist = []
    for _, r in bnd.iterrows():
        k = int(r.Interval)
        v = correct_vector(r.Ground_Truth, r.LLM_Output)
        idx = np.arange(len(v))
        is_post, is_within = position_classes(idx, k, h)
        if not is_post.any():
            continue
        post_rows.append(v[is_post].mean())
        ref = base_by_idx.get(len(v))
        if ref is not None:
            bpost.append(ref[is_post].mean())
        if k > h and is_within.any():               # no strictly internal position when k <= h
            within_rows.append(v[is_within].mean())
            if ref is not None:
                bwithin.append(ref[is_within].mean())
        if k in (8, 16):
            for dd in range(k):
                sel = (idx % k == dd) & (idx > 0)
                if sel.any():
                    dist.append((k, dd, v[sel].mean()))

    def m_ci(x):
        x = np.asarray(x)
        return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))

    out = {}
    for name, xs in [("base", base_rows), ("within", within_rows), ("post", post_rows)]:
        out[name], out[name + "_ci"] = m_ci(xs) if len(xs) > 1 else (np.nan, np.nan)
    out["base_at_post"] = float(np.mean(bpost)) if bpost else np.nan
    out["base_at_within"] = float(np.mean(bwithin)) if bwithin else np.nan
    out["h"] = h
    out["n_base_rows"], out["n_bounds_rows"] = len(base_rows), len(post_rows)
    out["n_within_rows"] = len(within_rows)
    dist_df = pd.DataFrame(dist, columns=["k", "d", "acc"]).groupby(["k", "d"]).acc.mean().reset_index() if dist else None
    return out, dist_df


def main():
    rows, dists = [], []
    for block, tasks in BLOCKS:
        for task in tasks:
            for label, tag in MODELS:
                d = load(task, tag)
                if d is None:
                    continue
                s, dist = summarize(d, horizon(task))
                if s is None:
                    continue
                rows.append(dict(block=block, task=task, model=label, **s))
                if dist is not None:
                    dist.insert(0, "model", label)
                    dist.insert(0, "task", task)
                    dist.insert(0, "block", block)
                    dists.append(dist)
    out = REPO / "results_clean" / "_summary"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(out / "per_position_values.csv", index=False)
    pd.concat(dists).round(4).to_csv(out / "per_position_distance.csv", index=False)
    print(f"Wrote {out / 'per_position_values.csv'} ({len(df)} task x model cells)")
    if "--print" in sys.argv:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_rows", 200)
        print(df[["block", "task", "model", "base", "base_at_post", "within", "post", "post_ci"]].to_string(index=False))


if __name__ == "__main__":
    main()
