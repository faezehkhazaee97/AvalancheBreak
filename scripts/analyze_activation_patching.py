#!/usr/bin/env python3
"""Analysis of scripts/run_activation_patching.py (reviewer question: where does the blind spot
enter?). For every model with a results/activation_patching/<tag>.csv:

  baselines   mean over trials of the post-marker log-odds (correct digit against the reset digit)
              and argmax accuracy in the clean (no-marker) and marked teacher-forced runs
  recovery    for every patch site and layer, (patched - marked) / (clean - marked) of the
              trial-mean log-odds, so 0 is the marked run and 1 is the clean run; the accuracy
              recovery is defined the same way

Sites: the input digit before each marker (pre_in), the input digit after it (post_in), every
input digit (all_in), and the answer digit written just before the marker (prev_out), patched at
one layer at a time and at every layer at once ("all").

Outputs: results_clean/_summary/activation_patching.csv,
         paper_tmlr/figures/patching.pdf (recovery by layer, one line per site, one panel per task),
         paper_tmlr/tables/patching.tex (baselines, all-layer recovery per site, best single layer)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH  # noqa: E402

MODELS = [("Qwen2.5 7B", "Qwen2.5-7B-Instruct"), ("Qwen2.5 32B", "Qwen2.5-32B-Instruct")]
SITES = [("pre_in", "input digit before the marker", "#1b9e77"), ("post_in", "input digit after the marker", "#d95f02"),
         ("all_in", "every input digit", "#7570b3"), ("prev_out", "answer digit before the marker", "#e7298a")]
TASKN = {"running_xor": "Running XOR", "running_sum_local": "Running sum (mod 4)"}


def main():
    rows, fig_data = [], {}
    for label, tag in MODELS:
        f = REPO / "results" / "activation_patching" / f"{tag}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["Layer"] = d.Layer.astype(str)
        for task, x in d.groupby("Task"):
            trials = sorted(set(x[x.Site == "clean"].Trial) & set(x[x.Site == "marked"].Trial))
            x = x[x.Trial.isin(trials)]
            m = x.groupby(["Site", "Layer"])[["logodds", "acc"]].mean()
            clean, marked = m.loc[("clean", "-1")], m.loc[("marked", "-1")]
            rows.append(dict(model=label, task=task, site="clean", layer="-1", n_trials=len(trials),
                             logodds=clean.logodds, acc=clean.acc, rec_logodds=1.0, rec_acc=1.0))
            rows.append(dict(model=label, task=task, site="marked", layer="-1", n_trials=len(trials),
                             logodds=marked.logodds, acc=marked.acc, rec_logodds=0.0, rec_acc=0.0))
            for site, _, _ in SITES:
                for (s, layer), r in m.iterrows():
                    if s != site:
                        continue
                    # only layers present for every trial
                    if (x[(x.Site == site) & (x.Layer == layer)].Trial.nunique()) < len(trials):
                        continue
                    rows.append(dict(model=label, task=task, site=site, layer=layer, n_trials=len(trials),
                                     logodds=r.logodds, acc=r.acc,
                                     rec_logodds=(r.logodds - marked.logodds) / (clean.logodds - marked.logodds),
                                     rec_acc=(r.acc - marked.acc) / (clean.acc - marked.acc) if clean.acc != marked.acc else float("nan")))
            fig_data[(label, task)] = True
    out = pd.DataFrame(rows).round(4)
    if out.empty:
        print("no results yet"); return
    out.to_csv(REPO / "results_clean" / "_summary" / "activation_patching.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[out.layer.isin(["-1", "all"])].to_string(index=False))

    # figure: one row per model, one panel per task; x = layer, y = log-odds recovery, line per site
    # figure only for models whose clean and marked runs differ enough to define a recovery (gap >= 2 nats)
    gap = {m: abs(float(out[(out.model == m) & (out.site == "clean")].logodds.mean() - out[(out.model == m) & (out.site == "marked")].logodds.mean())) for m, _ in MODELS}
    models = [m for m in MODELS if any(k[0] == m[0] for k in fig_data) and gap.get(m[0], 0) >= 2]
    if models:
        set_rc()
        fig, axes = plt.subplots(len(models), len(TASKN), figsize=(MULTI_COL_WIDTH, 2.1 * len(models)), squeeze=False)
        for i, (label, _) in enumerate(models):
            for j, task in enumerate(TASKN):
                ax = axes[i][j]
                for site, sname, col in SITES:
                    s = out[(out.model == label) & (out.task == task) & (out.site == site)]
                    per = s[s.layer != "all"].assign(L=lambda z: z.layer.astype(int)).sort_values("L")
                    if per.empty:
                        continue
                    ax.plot(per.L, per.rec_logodds, color=col, linewidth=1.4, label=sname)
                    a = s[s.layer == "all"]
                    if not a.empty:
                        ax.axhline(float(a.rec_logodds.iloc[0]), color=col, linewidth=0.8, linestyle=":")
                ax.axhline(0, color="black", linewidth=0.6); ax.axhline(1, color="black", linewidth=0.6, linestyle="--")
                ax.set_title(f"{label}: {TASKN[task]}", loc="left")
                ax.set_xlabel("patched layer"); ax.set_ylabel("log-odds recovery" if j == 0 else "")
        axes[0][-1].legend(frameon=False, loc="upper right", fontsize=6.8)
        fig.tight_layout()
        savefig(fig, REPO / "paper_tmlr" / "figures" / "patching.pdf")

    # table
    lines = ["\\begin{tabular}{ll cc cccc}", "\\toprule",
             "& & \\multicolumn{2}{c}{Post-marker acc.} & \\multicolumn{4}{c}{Recovery, all layers (best single layer)} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}",
             "Model & Task & Clean & Marked & Input before & Input after & All inputs & Answer before \\\\", "\\midrule"]
    for label, _ in MODELS:
        first = True
        for task in TASKN:
            s = out[(out.model == label) & (out.task == task)]
            if s.empty:
                continue
            cells = []
            for site, _, _ in SITES:
                z = s[s.site == site]
                a = z[z.layer == "all"]
                per = z[z.layer != "all"]
                if a.empty or per.empty:
                    cells.append("--"); continue
                b = per.loc[per.rec_logodds.idxmax()]
                cells.append(f"{float(a.rec_logodds.iloc[0]):.2f} ({float(b.rec_logodds):.2f} at {b.layer})")
            cl = float(s[s.site == "clean"].acc.iloc[0]); mk = float(s[s.site == "marked"].acc.iloc[0])
            lines.append(f"{label if first else ''} & {TASKN[task]} & {cl:.2f} & {mk:.2f} & " + " & ".join(cells) + " \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "patching.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/patching.tex")


if __name__ == "__main__":
    main()
