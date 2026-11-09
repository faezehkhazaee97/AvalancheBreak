"""Per-position accuracy on the ORIGINAL tasks: Base vs Bounds at positions right after a marker ('cut')
vs all other positions ('uncut'), all models, N<=32 (k=1 at N=64 is budget-capped) and k in {1,2,4,8,16}."""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
REPO = Path("/group/gquongrp/workspaces/baran/AvalancheBreak")
MODELS = {"Q7": "Qwen2.5-7B-Instruct", "Q14": "Qwen2.5-14B-Instruct", "Q32": "Qwen2.5-32B-Instruct", "Mis": "Mistral-7B-Instruct-v0.3",
          "Phi": "Phi-3.5-mini-instruct", "Q3.8": "Qwen3.8-27B", "G4": "gemma-4-31B-it"}
PL = ["cyclic_shift", "random_permutation", "conditional_shift"]
DEP = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]
GEM = REPO / "results_clean" / "gemini_pilot" / "gemini-flash-latest_POOLED_3trials.csv"

def load(task, tag):
    if tag is None:
        if not GEM.exists(): return None
        df = pd.read_csv(GEM, dtype=str); df = df[df.Task == task]
        if not len(df): return None
        for c in ("Interval", "Seq_Length", "Trial"): df[c] = df[c].astype(float).astype(int)
        return df
    files = [f for f in sorted((REPO / "results_clean" / task / "grid").glob(f"{tag}*.csv")) if "N128" not in f.name]
    if not files: return None
    df = pd.concat([pd.read_csv(f, dtype={"LLM_Output": str, "Ground_Truth": str, "Input_Given": str}) for f in files], ignore_index=True)
    axis = next((c for c in ("Alphabet_Size", "Modulus") if c in df.columns), None)
    return df.drop_duplicates(subset=["Method", "Interval", "Seq_Length", "Trial"] + ([axis] if axis else []))

def stats(df, task, Ns):
    first = 2 if task == "sliding_window_max" else 1   # positions whose rule involves a lookback
    acc = {"base": [0, 0], "cut": [0, 0], "uncut": [0, 0]}
    byk = {}
    for _, r in df.iterrows():
        N, k = int(r.Seq_Length), int(r.Interval)
        if N not in Ns or k == 3: continue
        truth = str(r.Ground_Truth).replace("|", ""); out = str(r.LLM_Output).replace("|", "") if pd.notna(r.LLM_Output) and str(r.LLM_Output) != "nan" else ""
        if len(truth) != N: continue
        if r.Method == "Base":
            for p in range(first, N):
                acc["base"][1] += 1; acc["base"][0] += (p < len(out) and out[p] == truth[p])
        else:
            cut = {j for j in range(N) if j % k == 0 and j > 0}
            if task == "sliding_window_max": cut |= {j for j in range(N) if j % k == 1 and j > 1 and k > 1}
            kk = byk.setdefault(k, {"cut": [0, 0], "uncut": [0, 0]})
            for p in range(first, N):
                key = "cut" if p in cut else "uncut"
                ok = (p < len(out) and out[p] == truth[p])
                acc[key][1] += 1; acc[key][0] += ok; kk[key][1] += 1; kk[key][0] += ok
    f = lambda a: a[0] / a[1] if a[1] else np.nan
    row = {k: round(f(v), 3) for k, v in acc.items()}
    row["n_cut"] = acc["cut"][1]
    for k in sorted(byk):
        row[f"k{k}_cut"] = round(f(byk[k]["cut"]), 2); row[f"k{k}_uncut"] = round(f(byk[k]["uncut"]), 2)
    return row

for Ns, label in [((4, 8, 16, 32), "N<=32"), ((64,), "N=64 only (k=1 cell is budget-capped)")]:
    print(f"\n==================== {label} ====================")
    rows = []
    for m, tag in list(MODELS.items()) + [("Gemini", None)]:
        for t in PL + DEP:
            df = load(t, tag)
            if df is None: continue
            r = stats(df, t, Ns); r.update(model=m, task=t); rows.append(r)
    df = pd.DataFrame(rows)
    df["uncut-base"] = (df.uncut - df.base).round(3); df["cut-base"] = (df.cut - df.base).round(3)
    cols = ["model", "task", "base", "uncut", "cut", "uncut-base", "cut-base", "n_cut"] + [c for c in df.columns if c.startswith("k")]
    print(df[cols].to_string())
