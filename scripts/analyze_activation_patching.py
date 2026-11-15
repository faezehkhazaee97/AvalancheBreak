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

MODELS = [("Qwen2.5 7B", "Qwen2.5-7B-Instruct"), ("Qwen2.5 32B", "Qwen2.5-32B-Instruct"),
          ("Qwen3.8 27B", "Qwen3.8-27B"), ("Gemma 4 31B", "gemma-4-31B-it")]
SITES = [("pre_in", "input digit before the marker", "#009E73"), ("post_in", "input digit after the marker", "#D55E00"),
         ("all_in", "every input digit", "#0072B2"), ("prev_out", "answer digit before the marker", "#CC79A7")]
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
    # figure: causal-tracing style heatmaps, one row per model, one panel per task; rows are the patch
    # sites, columns the patched layer, color the log-odds recovery; a separate last column is the
    # patch at every layer at once. Only models whose clean-marked gap is at least 2 nats are shown.
    gap = {m: abs(float(out[(out.model == m) & (out.site == "clean")].logodds.mean() - out[(out.model == m) & (out.site == "marked")].logodds.mean())) for m, _ in MODELS}
    models = [m for m in MODELS if any(k[0] == m[0] for k in fig_data) and gap.get(m[0], 0) >= 2]
    if models:
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        set_rc()
        cmap = LinearSegmentedColormap.from_list("rec", ["#f7f7f7", "#9ecae1", "#0072B2", "#08306b"])
        fig, axes = plt.subplots(len(models), len(TASKN), figsize=(MULTI_COL_WIDTH, 1.25 * len(models) + 0.5), squeeze=False)
        snames = {s: n for s, n, _ in SITES}
        short = {"pre_in": "input before marker", "post_in": "input after marker", "all_in": "every input", "prev_out": "answer before marker"}
        for i, (label, _) in enumerate(models):
            for j, task in enumerate(TASKN):
                ax = axes[i][j]
                sub = out[(out.model == label) & (out.task == task)]
                L = int(sub[sub.layer != "all"].layer.astype(int).max()) + 1 if not sub.empty else 0
                grid = np.full((len(SITES), L + 2), np.nan)
                for r, (site, _, _) in enumerate(SITES):
                    z = sub[sub.site == site]
                    for _, row in z[z.layer != "all"].iterrows():
                        grid[r, int(row.layer)] = row.rec_logodds
                    a = z[z.layer == "all"]
                    if not a.empty:
                        grid[r, L + 1] = float(a.rec_logodds.iloc[0])
                im = ax.imshow(np.ma.masked_invalid(np.clip(grid, 0, 1)), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
                ax.set_yticks(range(len(SITES))); ax.set_yticklabels([short[s] for s, _, _ in SITES] if j == 0 else [""] * len(SITES), fontsize=6)
                ticks = [0, 10, 20, 30, 40, 50]; ax.set_xticks(ticks + [L + 1]); ax.set_xticklabels([str(t) for t in ticks] + ["all"], fontsize=6)
                ax.set_title(f"{label}: {TASKN[task]}", fontsize=7.5); ax.grid(False)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(True)
                if i == len(models) - 1:
                    ax.set_xlabel("Patched layer")
        cax = fig.add_axes([0.92, 0.2, 0.012, 0.6]); cb = fig.colorbar(im, cax=cax)
        cb.set_label("Log-odds recovery (0 = marked run, 1 = clean run)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
        fig.subplots_adjust(left=0.17, right=0.9, top=0.93, bottom=0.1, hspace=0.6, wspace=0.08)
        savefig(fig, REPO / "paper_tmlr" / "figures" / "patching.pdf")

    # table
    def num(v):
        return "0.00" if abs(v) < 0.005 else f"${v:.2f}$"
    lines = ["\\begin{tabular}{ll cc c cccc}", "\\toprule",
             "& & \\multicolumn{2}{c}{Post-marker acc.} & Gap & \\multicolumn{4}{c}{Recovery, all layers (best single layer)} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){6-9}",
             "Model & Task & Clean & Marked & (nats) & Input before & Input after & All inputs & Answer before \\\\", "\\midrule"]
    for label, _ in MODELS:
        first = True
        for task in TASKN:
            s = out[(out.model == label) & (out.task == task)]
            if s.empty:
                continue
            cells = []
            gap = float(s[s.site == "clean"].logodds.iloc[0]) - float(s[s.site == "marked"].logodds.iloc[0])
            for site, _, _ in SITES:
                z = s[s.site == site]
                a = z[z.layer == "all"]
                per = z[z.layer != "all"]
                if a.empty or per.empty or gap < 2:      # ratios over a gap under 2 nats are not interpretable
                    cells.append("--"); continue
                b = per.loc[per.rec_logodds.idxmax()]
                cells.append(f"{num(float(a.rec_logodds.iloc[0]))} ({num(float(b.rec_logodds))} at {b.layer})")
            cl = float(s[s.site == "clean"].acc.iloc[0]); mk = float(s[s.site == "marked"].acc.iloc[0])
            lines.append(f"{label if first else ''} & {TASKN[task]} & {cl:.2f} & {mk:.2f} & {gap:.1f} & " + " & ".join(cells) + " \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "patching.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/patching.tex")


if __name__ == "__main__":
    main()
