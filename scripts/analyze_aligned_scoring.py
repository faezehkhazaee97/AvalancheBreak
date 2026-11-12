#!/usr/bin/env python3
"""Alignment-robust re-scoring: how much of the within-segment loss on dependency tasks is
cascading misalignment rather than local errors?

The paper's TAR compares prediction and truth index by index, so one dropped or inserted
character misaligns everything after it. Here each prediction is aligned to the truth by
minimum edit distance (substitution, insertion and deletion all cost 1; the alignment only uses
an insertion or deletion when that saves at least as many substitutions), and a truth position
counts as correct if it is aligned to an identical predicted character. The same scorer is
applied to marker and no-marker runs, then the per-position comparison of
scripts/analyze_per_position.py is repeated (post-marker vs within-segment, each against the
no-marker run at the same indices; N <= 64, k in {2,4,8,16}).

Output: results_clean/_summary/aligned_scoring.csv (one row per task x model x scorer).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS, LENGTHS, KS  # noqa: E402

TASKS = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]


def clean(s):
    return str(s).replace("|", "").replace(" ", "")


def index_vector(t, p):
    p = p[: len(t)]
    return np.array([i < len(p) and p[i] == c for i, c in enumerate(t)], dtype=float)


def aligned_vector(t, p):
    n, m = len(t), len(p)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ti = t[i - 1]
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ti != p[j - 1]))
    ok = np.zeros(n)
    i, j = n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (t[i - 1] != p[j - 1]):
            ok[i - 1] = t[i - 1] == p[j - 1]
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return ok


def summarize(d, vec):
    d = d[d.Seq_Length.astype(int).isin(LENGTHS)]
    base = d[d.Method == "Base"]
    bnd = d[(d.Method == "Cellular_Bounds") & d.Interval.astype(int).isin(KS)]
    def text(r):  # full generation where stored, else the stored (clipped) answer
        g = r.get("Raw_Model_Output", "")
        g = "" if pd.isna(g) else str(g).strip().split("\n")[0]
        return clean(g) if g else clean(r.LLM_Output)
    bvec = {}
    allb = []
    for _, r in base.iterrows():
        v = vec(clean(r.Ground_Truth), text(r))
        bvec.setdefault(len(v), []).append(v)
        allb.append(v.mean())
    ref = {n: np.mean(vs, axis=0) for n, vs in bvec.items()}
    post, within, bpost, bwithin = [], [], [], []
    for _, r in bnd.iterrows():
        k = int(r.Interval)
        v = vec(clean(r.Ground_Truth), text(r))
        idx = np.arange(len(v))
        is_post = (idx > 0) & (idx % k == 0)
        if not is_post.any() or len(v) not in ref:
            continue
        post.append(v[is_post].mean()); within.append(v[~is_post].mean())
        bpost.append(ref[len(v)][is_post].mean()); bwithin.append(ref[len(v)][~is_post].mean())
    return dict(base=np.mean(allb), post=np.mean(post), within=np.mean(within),
                post_gain=np.mean(post) - np.mean(bpost), within_gain=np.mean(within) - np.mean(bwithin))


def main():
    rows = []
    for task in TASKS:
        for label, tag in MODELS:
            if tag is None or str(tag).startswith("gemini"):
                continue
            d = load(task, tag)
            if d is None:
                continue
            for name, vec in [("index", index_vector), ("aligned", aligned_vector)]:
                rows.append(dict(task=task, model=label, scorer=name, **summarize(d, vec)))
            print(f"done {task} {label}", flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(REPO / "results_clean" / "_summary" / "aligned_scoring.csv", index=False)
    print(out.pivot_table(index=["task", "model"], columns="scorer", values=["within_gain", "post_gain"]).round(3).to_string())


if __name__ == "__main__":
    main()
