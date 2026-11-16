#!/usr/bin/env python3
"""Two follow-up analyses for the review response.

1. Wording test (tasks/common/wording.py). Does the post-marker blind spot depend on calling the
   bars "structural boundaries"? For running_xor and running_sum_local at Qwen2.5-32B,
   Qwen3.8-27B and Gemma-4-31B, the per-position comparison of scripts/analyze_per_position.py
   (post-marker and within-segment positions, each against the no-marker run of the same
   prompt variant at the same indices; N <= 64, k in {2,4,8,16}; index scoring) is repeated for
   the original, neutral and across wordings.
   Output: results_clean/_summary/wording_per_position.csv

2. Head-count dose response (scripts/run_head_ablation_v2.py with --top 10/50/200). At
   Qwen2.5-32B, random_permutation spaced letters and digits, A = 8, N = 64, k = 8: the share of
   the marker gain removed by blocking the top-n marker-attending heads, and by the random-head
   and sham-position controls, with paired 95% intervals on the drop.
   Output: results_clean/_summary/head_dose.csv

3. Single-trial check on the head ablations (--dose-outlier; writes no table). Finds the
   generations whose paired drop (markers minus intervention) is large in the control
   conditions, recomputes the share of the gain removed without the outlying trial for
   Table tab:ablation (spaced letters, N = 64, k in {2, 8}) and Table tab:dose (k = 8), prints
   that trial's outputs in every condition, and summarizes how the random-head controls are
   drawn (uniformly over all layers and heads except the top set, seeded by task, N and k;
   scripts/run_head_ablation_v2.py) against the layers of the top heads.
   Output: results_clean/_summary/head_dose_outlier.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, horizon  # noqa: E402
from analyze_aligned_scoring import summarize, index_vector  # noqa: E402

MODELS = [("Qwen2.5 32B", "Qwen2.5-32B-Instruct"), ("Qwen3.8 27B", "Qwen3.8-27B"), ("Gemma 4 31B", "gemma-4-31B-it")]
TASKS = ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]
VARIANTS = [("original", ""), ("neutral", "_neutral"), ("across", "_across"), ("worked", "_worked")]


def wording():
    rows = []
    for task in TASKS:
        for label, tag in MODELS:
            for v, suf in VARIANTS:
                d = load(task + suf, tag)
                if d is None:
                    continue
                r = summarize(d, index_vector, horizon(task))
                rows.append(dict(task=task, model=label, wording=v, **{k: round(float(x), 4) for k, x in r.items()}))
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results_clean" / "_summary" / "wording_per_position.csv", index=False)
    print(out.pivot_table(index=["task", "model"], columns="wording", values=["post_gain", "within_gain"])
          .round(2).to_string())
    print(out.pivot_table(index=["task", "model"], columns="wording", values=["post", "base"]).round(2).to_string())
    return out


def dose():
    src = {10: "Qwen2.5-32B-Instruct.csv", 50: "Qwen2.5-32B-Instruct_top50.csv", 200: "Qwen2.5-32B-Instruct_top200.csv"}
    rows = []
    for n, f in src.items():
        d = pd.read_csv(REPO / "results" / "ablation_v2" / f)
        for task in ["random_permutation_spaced", "random_permutation_digits"]:
            x = d[(d.Task == task) & (d.Seq_Length == 64)]
            base = x[x.Condition == "base"].TAR.mean()
            b = x[(x.Interval == 8)].pivot_table(index="Trial", columns="Condition", values="TAR")
            gain = b["bounds"].mean() - base
            for cond in ["ablate_top", "ctrl_random", "ctrl_sham"]:
                drop = b["bounds"] - b[cond]
                rows.append(dict(task=task, heads=n, condition=cond, base=round(base, 3),
                                 bounds=round(b["bounds"].mean(), 3), blocked=round(b[cond].mean(), 3),
                                 drop=round(drop.mean(), 3), ci=round(1.96 * drop.sem(), 3),
                                 share=round(drop.mean() / gain, 3) if gain > 0 else np.nan, trials=len(b)))
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results_clean" / "_summary" / "head_dose.csv", index=False)
    print(out.to_string(index=False))
    return out


ABL = REPO / "results" / "ablation_v2"
DOSE_FILES = {10: "Qwen2.5-32B-Instruct.csv", 50: "Qwen2.5-32B-Instruct_top50.csv", 200: "Qwen2.5-32B-Instruct_top200.csv"}
CONDS = ["ablate_top", "ctrl_random", "ctrl_sham"]


def _shares(x, ks, drop_trial=None):
    """Share of the marker gain removed per condition, as in make_table_ablation_v2.py (pooled over ks)
    and in dose() (paired over trials); drop_trial = (k, trial) removes that marker generation from every
    marker condition (the no-marker mean is kept: its inputs are drawn independently of k)."""
    base = x[x.Condition == "base"].TAR.mean()
    b = x[x.Interval.isin(ks) & (x.Condition != "base")]
    if drop_trial is not None:
        b = b[~((b.Interval == drop_trial[0]) & (b.Trial == drop_trial[1]))]
    m = b.groupby("Condition").TAR.mean()
    gain = m["bounds"] - base
    return {c: (m["bounds"] - m[c]) / gain for c in CONDS}, {c: m["bounds"] - m[c] for c in CONDS}, len(b[b.Condition == "bounds"])


def dose_outlier():
    task = "random_permutation_spaced"
    data = {n: pd.read_csv(ABL / f, dtype={c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")},
                           keep_default_na=False) for n, f in DOSE_FILES.items()}
    print("Large paired drops (|markers - intervention| >= 0.5), spaced letters:")
    hits = []
    for n, d in data.items():
        x = d[d.Task == task]
        p = x[x.Condition != "base"].pivot_table(index=["Seq_Length", "Interval", "Trial"], columns="Condition", values="TAR")
        for c in CONDS:
            dr = p.bounds - p[c]
            for key, v in dr[dr.abs() >= 0.5].items():
                hits.append((n, c) + key + (round(v, 3),))
    for h in hits:
        print(f"  {h[0]} heads, {h[1]}: N = {h[2]}, k = {h[3]}, trial {h[4]}: drop {h[5]:+.3f}")
    N, k, t = 64, 8, 14
    rows = []
    for n, d in data.items():
        x = d[(d.Task == task) & (d.Seq_Length == N)]
        for lab, ks in ([("tab:ablation (k in {2,8})", [2, 8])] if n == 10 else []) + [("tab:dose (k = 8)", [8])]:
            for excl in (None, (k, t)):
                sh, dr, ntr = _shares(x, ks, excl)
                for c in CONDS:
                    rows.append(dict(table=lab, heads=n, condition=c, without_trial=excl is not None, trials=ntr,
                                     drop=round(dr[c], 4), share=round(sh[c], 4)))
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results_clean" / "_summary" / "head_dose_outlier.csv", index=False)
    print(f"\nShare of the marker gain removed, spaced letters, N = {N}, with and without trial {t} at k = {k}:")
    print(out.pivot_table(index=["table", "heads", "condition"], columns="without_trial", values=["share", "drop"], sort=False)
          .round(3).to_string())
    print(f"\nOutputs of trial {t} (N = {N}, k = {k}) in every run:")
    for n, d in data.items():
        x = d[(d.Task == task) & (d.Seq_Length == N) & (d.Interval == k) & (d.Trial == t)]
        for _, r in x.iterrows():
            same = "" if r.LLM_Output != r.Ground_Truth else " (= truth)"
            print(f"  {n:>3} heads {r.Condition:<11} TAR {float(r.TAR):.3f} len {len(r.LLM_Output)}/{len(r.Ground_Truth)}  "
                  f"{r.Raw_Model_Output[:40]}...{same}")
    tr = data[10][(data[10].Task == task) & (data[10].Seq_Length == N) & (data[10].Interval == k) & (data[10].Trial == t)]
    tr = tr.set_index("Condition")
    truth, pred = tr.loc["bounds", "Ground_Truth"], tr.loc["ablate_top", "LLM_Output"]
    i = next((j for j, (a, b) in enumerate(zip(truth, pred)) if a != b), None)
    print(f"  first divergence from the truth at output index {i}: truth {truth[i - 2:i + 3]!r}, blocked runs "
          f"{pred[i - 2:i + 3]!r}; the rest of the output matches the truth shifted by one "
          f"({pred[i:] == truth[i + 1:]})")
    print("\nHead sets (layers of the top and random heads; Qwen2.5-32B has 64 layers x 40 heads):")
    for n, f in DOSE_FILES.items():
        h = pd.read_csv(ABL / f.replace(".csv", "_heads.csv"))
        h = h[(h.Task == task) & (h.Seq_Length == N) & (h.Interval == k)]
        for sset in ("top", "random"):
            L = h[h.Set == sset].Layer
            print(f"  {n:>3} heads, {sset:<6}: median layer {L.median():.0f}, quartiles {L.quantile(.25):.0f}-{L.quantile(.75):.0f}, "
                  f"range {L.min()}-{L.max()}, distinct layers {L.nunique()}, mean marker mass {h[h.Set == sset].Marker_Mass.mean():.3f}")


def fmt(x):
    return f"${x:+.2f}$"


def write_tables(w, d):
    tdir = REPO / "paper_tmlr" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{tabular}{ll cccc cccc}", "\\toprule",
             "& & \\multicolumn{4}{c}{Post-marker} & \\multicolumn{4}{c}{Within-segment} \\\\",
             "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}",
             "Task & Model & Orig. & Neutral & Across & + worked & Orig. & Neutral & Across & + worked \\\\", "\\midrule"]
    for task in TASKS:
        first = True
        for label, _ in MODELS:
            x = w[(w.task == task) & (w.model == label)].set_index("wording")
            if x.empty:
                continue
            g = lambda v, c: fmt(x.loc[v, c]) if v in x.index else "--"
            cells = [g(v, "post_gain") for v, _ in VARIANTS] + [g(v, "within_gain") for v, _ in VARIANTS]
            name = "\\task{" + task.replace("_", chr(92) + "_") + "}" if first else ""
            lines.append(f"{name} & {label} & " + " & ".join(cells) + " \\\\")
            first = False
        lines.append("\\addlinespace")
    lines[-1:] = ["\\bottomrule", "\\end{tabular}"]
    (tdir / "wording.tex").write_text("\n".join(lines) + "\n")
    lines = ["\\begin{tabular}{ll ccc}", "\\toprule",
             "Format & Heads blocked & Top $n$ & Random $n$ & Sham \\\\", "\\midrule"]
    for task, lab in [("random_permutation_spaced", "Spaced letters"), ("random_permutation_digits", "Digits")]:
        for n in (10, 50, 200):
            x = d[(d.task == task) & (d.heads == n)].set_index("condition")
            cells = [f"${int(round(100 * x.loc[c, 'share']))}\\%$ (${x.loc[c, 'drop']:+.3f} \\pm {x.loc[c, 'ci']:.3f}$)"
                     for c in ("ablate_top", "ctrl_random", "ctrl_sham")]
            lines.append(f"{lab if n == 10 else ''} & {n} & " + " & ".join(cells) + " \\\\")
        if task.endswith("spaced"):
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "head_dose.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/wording.tex and head_dose.tex")


if __name__ == "__main__":
    if "--dose-outlier" in sys.argv:
        dose_outlier()
    else:
        write_tables(wording(), dose())
