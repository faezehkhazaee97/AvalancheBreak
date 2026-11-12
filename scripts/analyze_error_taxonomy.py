#!/usr/bin/env python3
"""Two reviewer questions answered from existing runs.

1. Error taxonomy. Index scoring charges every position after a dropped or inserted character;
   edit-distance alignment charges only the slip itself. The difference between the index-scored
   and alignment-scored within-segment gain is therefore the part of the gain that comes from
   fewer alignment slips (desynchronization), and the remainder is fewer substitutions. Uses
   results_clean/_summary/aligned_scoring.csv (scripts/analyze_aligned_scoring.py).

2. Multi-step dependencies. sliding_window_max needs the two previous inputs, so a marker
   should blind the first TWO positions after it. Accuracy by distance after the marker
   (0, 1, 2, 3+) against the no-marker run at the same positions, k >= 4, N <= 64, four
   stronger models; running_xor (one-step) alongside for contrast.

Output: results_clean/_summary/error_taxonomy.csv, results_clean/_summary/blind_spot_by_distance.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, LENGTHS  # noqa: E402
from analyze_aligned_scoring import index_vector, clean  # noqa: E402

STRONG = ["Qwen2.5 14B", "Qwen2.5 32B", "Qwen3.8 27B", "Gemma 4 31B"]
TAGS = {"Qwen2.5 14B": "Qwen2.5-14B-Instruct", "Qwen2.5 32B": "Qwen2.5-32B-Instruct",
        "Qwen3.8 27B": "Qwen3.8-27B", "Gemma 4 31B": "gemma-4-31B-it"}


def taxonomy():
    a = pd.read_csv(REPO / "results_clean" / "_summary" / "aligned_scoring.csv")
    a = a[a.model.isin(STRONG)]
    p = a.pivot_table(index=["task", "model"], columns="scorer", values="within_gain").reset_index()
    p["desync_share"] = (p["index"] - p["aligned"]) / p["index"]
    p = p[p["index"] > 0.02]   # share is only meaningful where there is a gain to split
    p.round(3).to_csv(REPO / "results_clean" / "_summary" / "error_taxonomy.csv", index=False)
    print("within-segment gain: index-scored, alignment-scored, share attributable to fewer slips")
    print(p.round(2).to_string(index=False))
    print("median desync share over cells with a gain > 0.02:", round(p.desync_share.median(), 2),
          "| range", round(p.desync_share.min(), 2), "to", round(p.desync_share.max(), 2))


def by_distance():
    rows = []
    for task in ["running_xor", "sliding_window_max"]:
        for lab in STRONG:
            d = load(task, TAGS[lab])
            if d is None:
                continue
            d = d[d.Seq_Length.astype(int).isin(LENGTHS)]
            base = d[d.Method == "Base"]; bnd = d[(d.Method == "Cellular_Bounds") & (d.Interval.astype(int) >= 4)]
            ref = {}
            for _, r in base.iterrows():
                v = index_vector(clean(r.Ground_Truth), clean(r.LLM_Output)); ref.setdefault(len(v), []).append(v)
            ref = {n: np.mean(vs, axis=0) for n, vs in ref.items()}
            acc = {0: [], 1: [], 2: [], 3: []}; bacc = {0: [], 1: [], 2: [], 3: []}
            for _, r in bnd.iterrows():
                k = int(r.Interval); v = index_vector(clean(r.Ground_Truth), clean(r.LLM_Output))
                if len(v) not in ref:
                    continue
                idx = np.arange(len(v)); dist = idx % k
                for dd in (0, 1, 2, 3):
                    m = (idx >= k) & ((dist == dd) if dd < 3 else (dist >= 3))
                    if m.any():
                        acc[dd].append(v[m].mean()); bacc[dd].append(ref[len(v)][m].mean())
            for dd in (0, 1, 2, 3):
                rows.append(dict(task=task, model=lab, distance=("3+" if dd == 3 else str(dd)),
                                 acc=np.mean(acc[dd]), change=np.mean(acc[dd]) - np.mean(bacc[dd])))
    o = pd.DataFrame(rows).round(3)
    o.to_csv(REPO / "results_clean" / "_summary" / "blind_spot_by_distance.csv", index=False)
    print("\nchange vs no-marker run by distance after the marker (k >= 4):")
    print(o.pivot_table(index=["task", "model"], columns="distance", values="change").round(2).to_string())
    print("\nmean over the four models:")
    print(o.groupby(["task", "distance"]).change.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    taxonomy(); by_distance()
