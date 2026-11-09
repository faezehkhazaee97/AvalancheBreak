import sys, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
REPO = Path("/group/gquongrp/workspaces/baran/AvalancheBreak")
LENGTHS = [4, 8, 16, 32, 64]
MODELS = {"Q7": "Qwen2.5-7B-Instruct", "Q14": "Qwen2.5-14B-Instruct", "Q32": "Qwen2.5-32B-Instruct",
          "Mis": "Mistral-7B-Instruct-v0.3", "Phi": "Phi-3.5-mini-instruct", "Q3.8": "Qwen3.8-27B", "G4": "gemma-4-31B-it"}
PL = ["cyclic_shift", "random_permutation", "conditional_shift"]
DEP = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]
RESET = [t + "_reset" for t in DEP]

def load(task, tag):
    files = sorted((REPO / "results_clean" / task / "grid").glob(f"{tag}*.csv"))
    files = [f for f in files if "N128" not in f.name]
    if not files: return None
    df = pd.concat([pd.read_csv(f, dtype={"LLM_Output": str, "Ground_Truth": str, "Input_Given": str}) for f in files], ignore_index=True)
    axis = next((c for c in ("Alphabet_Size", "Modulus") if c in df.columns), None)
    keys = ["Method", "Interval", "Seq_Length", "Trial"] + ([axis] if axis else [])
    return df.drop_duplicates(subset=keys)

def delta(df):
    b, c = df[df.Method == "Base"].TAR, df[df.Method == "Cellular_Bounds"].TAR
    if len(b) < 2 or len(c) < 2: return (np.nan, np.nan, len(b), len(c), np.nan, np.nan)
    d = c.mean() - b.mean(); se = np.sqrt(b.sem()**2 + c.sem()**2)
    return (d, d / se if se > 0 else np.nan, len(b), len(c), b.mean(), c.mean())

data = {}
for t in PL + DEP + RESET:
    for m, tag in MODELS.items():
        df = load(t, tag)
        if df is not None: data[(t, m)] = df

print("=== A. pooled delta (N<=64, k!=3) all models incl. new ===")
rows = []
for (t, m), df in data.items():
    d = df[(df.Interval != 3) & (df.Seq_Length.isin(LENGTHS))]
    dl, z, nb, nc, bm, cm = delta(d)
    d3 = df[(df.Seq_Length.isin(LENGTHS))]
    dl3, z3, *_ = delta(d3)
    rows.append(dict(task=t, model=m, base=round(bm, 3), bounds=round(cm, 3), delta=round(dl, 3), z=round(z, 1), n_base=nb, n_bounds=nc,
                     delta_k3incl=round(dl3, 3), z_k3incl=round(z3, 1)))
A = pd.DataFrame(rows)
print(A.pivot(index="task", columns="model", values="delta").reindex(PL + DEP + RESET)[list(MODELS)].to_string())
print("\n-- z --"); print(A.pivot(index="task", columns="model", values="z").reindex(PL + DEP + RESET)[list(MODELS)].to_string())
print("\n-- base TAR --"); print(A.pivot(index="task", columns="model", values="base").reindex(PL + DEP + RESET)[list(MODELS)].to_string())
print("\n-- n_base --"); print(A.pivot(index="task", columns="model", values="n_base").reindex(PL + DEP + RESET)[list(MODELS)].to_string())
print("\n=== B. k=3 included vs excluded: cells whose |z|>=2 status flips or sign flips ===")
for _, r in A.iterrows():
    s1, s2 = abs(r.z) >= 2, abs(r.z_k3incl) >= 2
    if s1 != s2 or np.sign(r.delta) != np.sign(r.delta_k3incl):
        print(f"  {r.task:28s} {r.model:5s} excl: {r.delta:+.3f} (z={r.z:+.1f})  incl: {r.delta_k3incl:+.3f} (z={r.z_k3incl:+.1f})")

print("\n=== C. per-density delta (Bounds@k - Base), dependency originals vs resets; N<=32 only (avoids k=1 cap) ===")
for m in ["Q7", "Q32", "Q3.8", "G4"]:
    print(f"--- {m}")
    out = {}
    for t in DEP + RESET:
        if (t, m) not in data: continue
        df = data[(t, m)]; df = df[df.Seq_Length.isin([4, 8, 16, 32])]
        base = df[df.Method == "Base"].TAR
        rec = {}
        for k in [1, 2, 4, 8, 16]:
            c = df[(df.Method == "Cellular_Bounds") & (df.Interval == k)].TAR
            # match base to same lengths as bounds cells exist for this k
            lens = df[(df.Method == "Cellular_Bounds") & (df.Interval == k)].Seq_Length.unique()
            bb = df[(df.Method == "Base") & (df.Seq_Length.isin(lens))].TAR
            if len(c) > 1: rec[f"k={k}"] = f"{c.mean()-bb.mean():+.3f}"
        out[t] = rec
    print(pd.DataFrame(out).T.to_string())

print("\n=== C2. same, all N<=64 (k=1 at N=64 is the capped cell) ===")
for m in ["Q32", "Q3.8", "G4"]:
    print(f"--- {m}")
    out = {}
    for t in DEP + RESET:
        if (t, m) not in data: continue
        df = data[(t, m)]; df = df[df.Seq_Length.isin(LENGTHS)]
        rec = {}
        for k in [1, 2, 4, 8, 16]:
            c = df[(df.Method == "Cellular_Bounds") & (df.Interval == k)]
            lens = c.Seq_Length.unique()
            bb = df[(df.Method == "Base") & (df.Seq_Length.isin(lens))].TAR
            if len(c) > 1: rec[f"k={k}"] = f"{c.TAR.mean()-bb.mean():+.3f}"
        out[t] = rec
    print(pd.DataFrame(out).T.to_string())

print("\n=== D. Reset-Bounds TAR minus Original-Bounds TAR at matched k, N<=32 ===")
for m in ["Q7", "Q32", "Q3.8", "G4"]:
    print(f"--- {m}")
    out = {}
    for t in DEP:
        if (t, m) not in data or (t + "_reset", m) not in data: continue
        o = data[(t, m)]; r = data[(t + "_reset", m)]
        o = o[o.Seq_Length.isin([4, 8, 16, 32])]; r = r[r.Seq_Length.isin([4, 8, 16, 32])]
        rec = {"Base(orig->reset)": f"{o[o.Method=='Base'].TAR.mean():.3f}->{r[r.Method=='Base'].TAR.mean():.3f}"}
        for k in [1, 2, 4, 8, 16]:
            oc = o[(o.Method == "Cellular_Bounds") & (o.Interval == k)].TAR; rc = r[(r.Method == "Cellular_Bounds") & (r.Interval == k)].TAR
            if len(oc) > 1 and len(rc) > 1: rec[f"k={k}"] = f"{oc.mean():.2f}->{rc.mean():.2f}"
        out[t] = rec
    print(pd.DataFrame(out).T.to_string())

print("\n=== E. per-length delta, Q32 dependency + Q3.8/G4 all tasks (k!=3) ===")
for m in ["Q32", "Q3.8", "G4"]:
    print(f"--- {m}")
    out = {}
    for t in PL + DEP + RESET:
        if (t, m) not in data: continue
        df = data[(t, m)]; df = df[df.Interval != 3]
        rec = {}
        for N in LENGTHS + [128, 256]:
            d = df[df.Seq_Length == N]
            dl, z, nb, nc, *_ = delta(d)
            if not np.isnan(dl): rec[f"N={N}"] = f"{dl:+.3f}({z:+.1f})"
        out[t] = rec
    print(pd.DataFrame(out).T.to_string())

print("\n=== F. reset-instruction compliance at distinguishing post-marker positions (Bounds rows, k!=3, N<=64) ===")
def orig_rule(task, x, M):
    n = len(x); y = []
    for i in range(n):
        if task == "running_xor": y.append(x[i] if i == 0 else str(int(x[i]) ^ int(x[i-1])))
        elif task == "copy_previous": y.append(x[i] if i == 0 else x[i-1])
        elif task == "running_sum_local": y.append(x[i] if i == 0 else str((int(x[i]) + int(x[i-1])) % int(M)))
        elif task == "sliding_window_max": y.append(max(x[max(0, i-2):i+1]))
    return "".join(y)
rows = []
for m in ["Q7", "Q32", "Q3.8", "G4"]:
    for t in DEP:
        if (t + "_reset", m) not in data: continue
        df = data[(t + "_reset", m)]
        df = df[(df.Method == "Cellular_Bounds") & (df.Interval != 3) & (df.Seq_Length.isin(LENGTHS))]
        fr = fo = fn = tot = 0; ntr = 0
        by_k = {}
        for _, r in df.iterrows():
            N, k = int(r.Seq_Length), int(r.Interval)
            x = str(r.Input_Given).replace("|", ""); truth = str(r.Ground_Truth).replace("|", "")
            out = str(r.LLM_Output).replace("|", "") if pd.notna(r.LLM_Output) else ""
            if len(x) != N: continue
            M = r.Modulus if "Modulus" in df.columns else None
            o = orig_rule(t, x, M)
            pos = [j for j in range(N) if j % k == 0 and j > 0]
            if t == "sliding_window_max": pos += [j for j in range(N) if j % k == 1 and j > 1 and k > 1]
            for p in pos:
                if p >= len(truth) or p >= len(o) or truth[p] == o[p]: continue
                tot += 1; c = out[p] if p < len(out) else None
                kk = by_k.setdefault(k, [0, 0, 0])
                kk[2] += 1
                if c == truth[p]: fr += 1; kk[0] += 1
                elif c == o[p]: fo += 1; kk[1] += 1
                else: fn += 1
        if tot:
            rows.append(dict(model=m, task=t + "_reset", n_pos=tot, follows_reset=round(fr/tot, 2), follows_carryover=round(fo/tot, 2), neither=round(fn/tot, 2),
                             by_k=" ".join(f"k{k}:{v[0]/v[2]:.2f}/{v[1]/v[2]:.2f}" for k, v in sorted(by_k.items()))))
print(pd.DataFrame(rows).to_string())

print("\n=== G. Base TAR, original prompt vs reset prompt (same inputs, no markers), N<=64 ===")
for m in ["Q7", "Q32", "Q3.8", "G4"]:
    for t in DEP:
        if (t, m) in data and (t + "_reset", m) in data:
            o = data[(t, m)]; r = data[(t + "_reset", m)]
            ob = o[(o.Method == "Base") & o.Seq_Length.isin(LENGTHS)].TAR; rb = r[(r.Method == "Base") & r.Seq_Length.isin(LENGTHS)].TAR
            d = rb.mean() - ob.mean(); se = np.sqrt(ob.sem()**2 + rb.sem()**2)
            print(f"  {m:5s} {t:20s} base orig {ob.mean():.3f} (n={len(ob)})  base reset {rb.mean():.3f} (n={len(rb)})  diff {d:+.3f} z={d/se:+.1f}")

print("\n=== H. k=1 truncation at N=64: share of Bounds outputs shorter than truth (local models) ===")
for (t, m), df in data.items():
    d = df[(df.Method == "Cellular_Bounds") & (df.Interval == 1) & (df.Seq_Length == 64)]
    if len(d):
        short = (d.LLM_Output.fillna("").str.replace("|", "").str.len() < d.Ground_Truth.str.replace("|", "").str.len()).mean()
        print(f"  {t:28s} {m:5s} n={len(d):4d} truncated={short:.2f} TAR={d.TAR.mean():.3f}")
