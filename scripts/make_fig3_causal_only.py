#!/usr/bin/env python3
"""Figure 3 (main paper, ICLR restructure): standalone causal figure.
Formerly panel (b) of the combined dip-mechanism-and-causal figure --
split out on its own now that panel (a) (the mechanistic half) has moved
into Figure 2 alongside the rest of the attention-sink evidence.

Causal test: for four dependency tasks, does genuinely severing the
cross-position dependency at each boundary (the "reset" variant) shrink
or reverse the harm boundaries otherwise cause? Qwen2.5-32B-Instruct.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH

set_rc()

CAT_ORANGE = "#eb6834"
GRAY = "#78776f"

fig, ax_b = plt.subplots(1, 1, figsize=(MULTI_COL_WIDTH * 0.85, 3.8))

tasks_deltas = {
    "running_xor": (0.015, 0.126),
    "running_sum_local": (0.011, 0.110),
    "sliding_window_max": (0.043, 0.112),
    "copy_previous": (-0.043, -0.107),
}
labels = list(tasks_deltas.keys())
orig = [tasks_deltas[t][0] for t in labels]
reset = [tasks_deltas[t][1] for t in labels]

x = np.arange(len(labels))
width = 0.35
ax_b.bar(x - width / 2, orig, width, label="Original (visual only)", color=GRAY)
ax_b.bar(x + width / 2, reset, width, label="Reset (dependency severed)", color=CAT_ORANGE)
ax_b.axhline(0, color="black", linewidth=0.8)
ax_b.set_xticks(x)
ax_b.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
ax_b.set_ylabel("Accuracy change from adding boundaries\n(TAR delta: + helps, &minus; hurts)".replace("&minus;", "−"), fontsize=8)
ax_b.set_title("Severing the dependency shrinks or reverses the harm\n-- except copy_previous (rule-switching side effect, see text)", fontsize=8.5)
ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.48), ncol=2, frameon=False, fontsize=7.5)

fig.suptitle("Causal confirmation: is the dependency itself the reason boundaries hurt? (Qwen2.5-32B-Instruct)", fontsize=9, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.subplots_adjust(bottom=0.4)

OUT = REPO_ROOT / "figs" / "fig3_causal_only.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
