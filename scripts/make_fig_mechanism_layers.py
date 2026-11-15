#!/usr/bin/env python3
"""Where the blind spot acts, layer by layer, at Qwen2.5-32B (teacher-forced, N = 32, k = 4).

Top row, per task: activation patching (scripts/run_activation_patching.py) as a site-by-layer map of
log-odds recovery, 0 = marked run, 1 = clean run. Bottom row, per task: attention from the step that
predicts each answer character to the input character before it (scripts/run_attention_crossing.py),
strongest head per layer, for the clean run at the post-marker indices, the marked run at its
within-segment steps, and the marked run at its post-marker steps. Shared layer axis, so the band of
lookback heads (layers 35 to 45) can be read against the band where patching recovers the prediction.

Output: figs/fig_mechanism_layers.pdf
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH  # noqa: E402

TASKS = [("running_xor", "Running XOR"), ("running_sum_local", "Running sum (mod 4)")]
SITES = [("pre_in", "input before marker"), ("post_in", "input after marker"), ("all_in", "every input"), ("prev_out", "answer before marker")]
ROWS = [("clean", "post", "no markers"), ("marked", "within", "markers, within segment"), ("marked", "post", "markers, after marker")]
REC = LinearSegmentedColormap.from_list("rec", ["#f7f7f7", "#9ecae1", "#0072B2", "#08306b"])
ATT = LinearSegmentedColormap.from_list("att", ["#f7f7f7", "#fdd0a2", "#D55E00", "#7f2704"])


def main():
    set_rc()
    pat = pd.read_csv(REPO / "results_clean" / "_summary" / "activation_patching.csv")
    pat = pat[(pat.model == "Qwen2.5 32B") & (pat.layer != "all") & pat.site.isin([s for s, _ in SITES])]
    att = pd.read_csv(REPO / "results" / "attention_crossing" / "Qwen2.5-32B-Instruct.csv")
    att = att.groupby(["Task", "Run", "PosType", "Layer"]).max_head_prev_in.mean().reset_index()
    L = int(att.Layer.max()) + 1
    fig, axes = plt.subplots(2, 2, figsize=(MULTI_COL_WIDTH, 3.1), sharex=True)
    for j, (task, tname) in enumerate(TASKS):
        g = np.full((len(SITES), L), np.nan)
        for r, (site, _) in enumerate(SITES):
            for _, row in pat[(pat.task == task) & (pat.site == site)].iterrows():
                g[r, int(row.layer)] = row.rec_logodds
        ax = axes[0][j]
        im1 = ax.imshow(np.clip(g, 0, 1), cmap=REC, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(SITES))); ax.set_yticklabels([n for _, n in SITES] if j == 0 else [""] * len(SITES))
        ax.set_title(f"({'ab'[j]}) {tname}: patching")
        a = np.full((len(ROWS), L), np.nan)
        for r, (run, ptype, _) in enumerate(ROWS):
            x = att[(att.Task == task) & (att.Run == run) & (att.PosType == ptype)].set_index("Layer").max_head_prev_in
            a[r, x.index.values] = x.values
        ax = axes[1][j]
        im2 = ax.imshow(a, cmap=ATT, vmin=0, vmax=0.6, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([n for _, _, n in ROWS] if j == 0 else [""] * len(ROWS))
        ax.set_title(f"({'cd'[j]}) {tname}: attention")
        ax.set_xlabel("Layer")
        for axx in (axes[0][j], axes[1][j]):
            axx.grid(False); axx.set_xticks([0, 10, 20, 30, 40, 50, 60])
            for sp in ("top", "right"):
                axx.spines[sp].set_visible(True)
    fig.subplots_adjust(left=0.17, right=0.89, top=0.93, bottom=0.12, hspace=0.42, wspace=0.06)
    for axx, im, lab in [(axes[0][1], im1, "Log-odds recovery"), (axes[1][1], im2, "Attention")]:
        p = axx.get_position()   # colorbar as tall as its row of panels
        cb = fig.colorbar(im, cax=fig.add_axes([0.905, p.y0, 0.011, p.height])); cb.set_label(lab)
    savefig(fig, REPO / "figs" / "fig_mechanism_layers.pdf")


if __name__ == "__main__":
    main()
