#!/usr/bin/env python3
"""Analysis of scripts/run_patching_aligned.py (reviewer question: does the patching result
survive a donor that is position-aligned with the marked run?).

The donor is the across-wording marked run: same task, same input, same ground-truth answer,
markers in the same places, so every relative distance among input and answer tokens matches the
recipient and the step after a marker is read at the output bar in both. Recovery at a site is
(patched - recipient) / (donor - recipient) on the trial-mean post-marker log-odds, so 0 is the
original marked run and 1 is the across-wording run.

The comparison this checks is the one in Table 27, which patches from the no-marker run and so
carries a progressive index shift and a different query token. If the ordering of sites is the
same here, neither confound produced it.

Outputs: results_clean/_summary/patching_aligned.csv
         paper_tmlr/tables/patching_aligned.tex
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SITEN = {"all_in": "Every input digit", "post_in": "Input after the marker",
         "pre_in": "Input before the marker", "prev_out": "Answer already written"}
TASKN = {"running_xor": "Running XOR", "running_sum_local": "Running sum (mod 4)"}
MODELS = [("Qwen2.5 32B", "Qwen2.5-32B-Instruct")]


def main():
    frames = []
    for label, tag in MODELS:
        f = REPO / "results" / "patching_aligned" / f"{tag}.csv"
        if not f.exists():
            print(f"no results for {label}")
            continue
        d = pd.read_csv(f)
        d.insert(0, "model", label)
        frames.append(d)
    if not frames:
        print("nothing to analyse yet")
        return
    d = pd.concat(frames, ignore_index=True)

    # mean over trials, per task/site/layer
    g = d.groupby(["model", "Task", "Site", "Layer"], dropna=False).agg(
        logodds=("logodds", "mean"), acc=("acc", "mean"), n=("Trial", "nunique"),
        offset=("prefix_offset", "max")).reset_index()

    rows = []
    for (model, task), sub in g.groupby(["model", "Task"]):
        base = sub[sub.Site == "recipient"]
        don = sub[sub.Site == "donor"]
        if base.empty or don.empty:
            continue
        b_lo, d_lo = float(base.logodds.iloc[0]), float(don.logodds.iloc[0])
        b_ac, d_ac = float(base.acc.iloc[0]), float(don.acc.iloc[0])
        gap = d_lo - b_lo
        for _, r in sub.iterrows():
            if r.Site in ("recipient", "donor"):
                continue
            rows.append(dict(model=model, task=task, site=r.Site, layer=r.Layer,
                             logodds=r.logodds, acc=r.acc, n_trials=r.n, prefix_offset=r.offset,
                             recipient_logodds=b_lo, donor_logodds=d_lo, gap=gap,
                             rec_logodds=(r.logodds - b_lo) / gap if gap else float("nan"),
                             rec_acc=(r.acc - b_ac) / (d_ac - b_ac) if d_ac != b_ac else float("nan")))
    o = pd.DataFrame(rows).round(4)
    out = REPO / "results_clean" / "_summary" / "patching_aligned.csv"
    o.to_csv(out, index=False)
    print(f"wrote {out} ({len(o)} rows)")

    pd.set_option("display.width", 220)
    allr = o[o.layer.astype(str) == "all"]
    print("\n=== all-layer recovery, position-aligned donor ===")
    print(allr.pivot_table(index="site", columns="task", values="rec_logodds").round(3).to_string())
    print("\nclean--marked gap (nats):")
    print(o.groupby("task").agg(recipient=("recipient_logodds", "first"),
                                donor=("donor_logodds", "first"), gap=("gap", "first")).round(2).to_string())

    lines = ["\\begin{tabular}{l cc cc}", "\\toprule",
             "& \\multicolumn{2}{c}{Log-odds recovery} & \\multicolumn{2}{c}{Best single layer} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Patched site & " + " & ".join(TASKN[t] for t in TASKN) + " & "
             + " & ".join(TASKN[t] for t in TASKN) + " \\\\", "\\midrule"]
    for site in ("all_in", "post_in", "pre_in", "prev_out"):
        cells, peaks = [], []
        for task in TASKN:
            a = allr[(allr.task == task) & (allr.site == site)]
            cells.append(f"{100 * a.rec_logodds.iloc[0]:.0f}\\%" if len(a) else "--")
            per = o[(o.task == task) & (o.site == site) & (o.layer.astype(str) != "all")].copy()
            if len(per):
                per["L"] = per.layer.astype(int)
                best = per.loc[per.rec_logodds.idxmax()]
                peaks.append(f"{100 * best.rec_logodds:.0f}\\% (L{int(best.L)})")
            else:
                peaks.append("--")
        lines.append(f"{SITEN[site]} & " + " & ".join(cells + peaks) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    t = REPO / "paper_tmlr" / "tables" / "patching_aligned.tex"
    t.write_text("\n".join(lines) + "\n")
    print(f"wrote {t}")


if __name__ == "__main__":
    main()
