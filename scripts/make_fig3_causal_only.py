#!/usr/bin/env python3
"""Figure 3: causal test with reset variants (Qwen2.5-32B-Instruct).

For four dependency tasks, the pooled change in token accuracy when markers are added,
with the dependency intact (original task) and severed (reset variant: the recurrence
restarts at every marker). Bars: pooled delta; whiskers: 95% confidence interval.

Values come from results_clean/_summary/split_heatmap_values.csv, which
scripts/make_fig_split_heatmap.py writes (same pooling as Figure 1), so run that first.
No title: the paper's caption carries the message.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH  # noqa: E402

MODEL = "Qwen2.5 32B"
TASKS = ["running_xor", "running_sum_local", "sliding_window_max", "copy_previous"]
CAT_ORANGE = "#eb6834"
GRAY = "#78776f"

vals = pd.read_csv(REPO_ROOT / "results_clean" / "_summary" / "split_heatmap_values.csv")
vals = vals[vals.model == MODEL].set_index("task")


def delta_ci(task):
    r = vals.loc[task]
    se = abs(r.delta / r.z) if r.z else 0.0
    return r.delta, 1.96 * se


orig = [delta_ci(t) for t in TASKS]
reset = [delta_ci(f"{t}_reset") for t in TASKS]

set_rc()
fig, ax = plt.subplots(1, 1, figsize=(MULTI_COL_WIDTH * 0.85, 3.2))
x = np.arange(len(TASKS))
width = 0.35
err_kw = dict(ecolor="#333333", elinewidth=0.8, capsize=2)
ax.bar(x - width / 2, [d for d, _ in orig], width, yerr=[c for _, c in orig],
       label="Dependency intact", color=GRAY, error_kw=err_kw)
ax.bar(x + width / 2, [d for d, _ in reset], width, yerr=[c for _, c in reset],
       label="Dependency severed (reset)", color=CAT_ORANGE, error_kw=err_kw)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(TASKS, rotation=20, ha="right", fontsize=7.5)
ax.set_ylabel("TAR change with markers\n(Bounds − Base)", fontsize=8)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=False, fontsize=7.5)
fig.tight_layout()
fig.subplots_adjust(bottom=0.36)

OUT = REPO_ROOT / "figs" / "fig3_causal_only.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Wrote {OUT}")
for t, (d0, c0), (d1, c1) in zip(TASKS, orig, reset):
    print(f"  {t:20s} intact {d0:+.3f} ±{c0:.3f}   severed {d1:+.3f} ±{c1:.3f}")
