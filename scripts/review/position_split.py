"""Within-segment (rule-unchanged) positions vs reset positions: does the reset variant's extra gain
come from positions whose rule changed (trivial copies) or from positions whose rule is identical?"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
REPO = Path("/group/gquongrp/workspaces/baran/AvalancheBreak")
MODELS = {"Q7": "Qwen2.5-7B-Instruct", "Q32": "Qwen2.5-32B-Instruct", "Q3.8": "Qwen3.8-27B", "G4": "gemma-4-31B-it"}
DEP = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]

def load(task, tag):
    files = [f for f in sorted((REPO / "results_clean" / task / "grid").glob(f"{tag}*.csv")) if "N128" not in f.name]
    if not files: return None
    df = pd.concat([pd.read_csv(f, dtype={"LLM_Output": str, "Ground_Truth": str, "Input_Given": str}) for f in files], ignore_index=True)
    axis = next((c for c in ("Alphabet_Size", "Modulus") if c in df.columns), None)
    return df.drop_duplicates(subset=["Method", "Interval", "Seq_Length", "Trial"] + ([axis] if axis else []))

def posacc(df, task, which, Ns=(4, 8, 16, 32), ks=(2, 4, 8, 16)):
    """mean per-position accuracy over positions of type `which` in {'nonreset','reset'} for Bounds rows at k in ks,
    and for Base rows at the same positions (using k of the matched cell is impossible for Base; use all k's position sets)."""
    hits = tot = 0
    for _, r in df.iterrows():
        N, k = int(r.Seq_Length), int(r.Interval)
        if N not in Ns: continue
        truth = str(r.Ground_Truth).replace("|", ""); out = str(r.LLM_Output).replace("|", "") if pd.notna(r.LLM_Output) else ""
        if len(truth) != N: continue
        if r.Method == "Cellular_Bounds":
            if k not in ks: continue
            rs = {j for j in range(N) if j % k == 0 and j > 0}
            if task == "sliding_window_max": rs |= {j for j in range(N) if j % k == 1 and j > 1}
            pos = rs if which == "reset" else set(range(1, N)) - rs
            for p in pos:
                tot += 1; hits += (p < len(out) and out[p] == truth[p])
    return hits / tot if tot else np.nan

rows = []
for m, tag in MODELS.items():
    for t in DEP:
        o = load(t, tag); r = load(t + "_reset", tag)
        if o is None or r is None: continue
        o = o[o.Interval != 3]; r = r[r.Interval != 3]
        # Base per-position accuracy at positions j>=1 (no markers; same for both position types) under each prompt
        def base_acc(df):
            h = n = 0
            for _, x in df[(df.Method == "Base") & df.Seq_Length.isin([4, 8, 16, 32])].iterrows():
                truth = str(x.Ground_Truth); out = str(x.LLM_Output) if pd.notna(x.LLM_Output) else ""
                for p in range(1, len(truth)):
                    n += 1; h += (p < len(out) and out[p] == truth[p])
            return h / n
        rows.append(dict(model=m, task=t,
                         base_orig=round(base_acc(o), 3), base_reset=round(base_acc(r), 3),
                         orig_bounds_nonreset=round(posacc(o, t, "nonreset"), 3), reset_bounds_nonreset=round(posacc(r, t, "nonreset"), 3),
                         orig_bounds_resetpos=round(posacc(o, t, "reset"), 3), reset_bounds_resetpos=round(posacc(r, t, "reset"), 3)))
df = pd.DataFrame(rows)
df["d_nonreset(reset-orig)"] = (df.reset_bounds_nonreset - df.orig_bounds_nonreset).round(3)
df["d_resetpos(reset-orig)"] = (df.reset_bounds_resetpos - df.orig_bounds_resetpos).round(3)
print("Per-position accuracy, N<=32, k in {2,4,8,16}. 'nonreset' = positions whose rule is identical in both variants.")
print(df.to_string())
