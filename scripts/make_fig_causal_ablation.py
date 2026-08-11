#!/usr/bin/env python3
"""Figure 8: Causal ablation figure (figs/causal_ablation.pdf). Left: TAR
vs. N for Base / Cellular_Bounds / ablated, Qwen2.5-7B. Right: % of
Cellular's benefit erased by ablation, Qwen2.5-7B vs. Qwen2.5-32B plotted
directly side by side (previously this scale comparison was only stated
in the caption text, not shown in the figure itself -- now both models'
bars are in the same panel so the reader doesn't have to cross-reference
Figure 1c's numbers from memory). 32B also gets an N=128 bar (7B was
never run at that length, so no paired bar there). Replaces old Figure 20.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, set_power_of_2_xticks, power_of_2_label

set_rc()

SRC_7B = REPO_ROOT / "results" / "Qwen2.5-7B-Instruct" / "causal_ablation_comparison.csv"
print(f"Provenance -- 7B source: {SRC_7B.relative_to(REPO_ROOT)}")
df7 = pd.read_csv(SRC_7B)

# 32B: same computation Figure 1c uses (causal_ablation_grid.csv vs. the
# thermodynamics_grid_templated_v2.csv reference), re-derived here so this
# figure doesn't silently drift from Figure 1c if either is regenerated.
RESULTS_32B = REPO_ROOT / "results" / "Qwen2.5-32B-Instruct"
abl32 = pd.read_csv(RESULTS_32B / "causal_ablation_grid.csv")
base_grid32 = pd.read_csv(RESULTS_32B / "thermodynamics_grid_templated_v2.csv")
lengths32 = sorted(abl32.Seq_Length.unique())
ref32 = base_grid32[(base_grid32.Alphabet_Size == 2) & (base_grid32.Seq_Length.isin(lengths32))]
base32 = ref32[ref32.Method == "Base"].groupby("Seq_Length")["TAR"].mean().reindex(lengths32)
normal32 = ref32[(ref32.Method == "Cellular_Bounds") & (ref32.Interval == 2)].groupby("Seq_Length")["TAR"].mean().reindex(lengths32)
ablated32 = abl32.groupby("Seq_Length")["TAR"].mean().reindex(lengths32)
pct32 = (normal32 - ablated32) / (normal32 - base32) * 100
print(f"Provenance -- 32B source: {(RESULTS_32B / 'causal_ablation_grid.csv').relative_to(REPO_ROOT)}, "
      f"{(RESULTS_32B / 'thermodynamics_grid_templated_v2.csv').relative_to(REPO_ROOT)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH, 3.0))

ax1.plot(df7.Seq_Length, df7.Base_TAR, color="black", linestyle="--", marker="o", markersize=4, label="Base")
ax1.plot(df7.Seq_Length, df7.Cellular_Normal_TAR, color="#1b9e77", linestyle="-", marker="s", markersize=4, label="Cellular_Bounds")
ax1.plot(df7.Seq_Length, df7.Cellular_Ablated_TAR, color="#d95f02", linestyle="-", marker="^", markersize=4, label="Cellular_Bounds (ablated)")
ax1.set_xlabel("Sequence length N")
ax1.set_ylabel("TAR")
set_power_of_2_xticks(ax1, df7.Seq_Length.unique(), fontsize=6.5)
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower left", frameon=False, fontsize=6.5)
ax1.set_title("Accuracy (Qwen2.5-7B)")

# Right panel: grouped bars, 7B vs 32B, at the shared lengths (8/16/32/64),
# plus a lone 32B bar at N=128 (7B has no data there).
shared_N = [8, 16, 32, 64]
x = np.arange(len(shared_N))
width = 0.35
vals7 = [df7.set_index("Seq_Length").loc[n, "Pct_of_Cellular_Benefit_Erased"] for n in shared_N]
vals32 = [pct32.loc[n] for n in shared_N]

b1 = ax2.bar(x - width / 2, vals7, width, color="#d95f02", label="Qwen2.5-7B")
b2 = ax2.bar(x + width / 2, vals32, width, color="#2a78d6", label="Qwen2.5-32B")
# N=128, 32B only -- placed as an extra bar to the right with a visible gap
gap_x = len(shared_N) + 0.3
ax2.bar([gap_x], [pct32.loc[128]], width, color="#2a78d6", hatch="//", edgecolor="white")
ax2.text(gap_x, pct32.loc[128] + 1.5, "N=128\n(32B only)", ha="center", fontsize=5.5)

for xi, y in zip(x - width / 2, vals7):
    ax2.text(xi, y + 1.5, f"{y:.0f}%", ha="center", fontsize=6)
for xi, y in zip(x + width / 2, vals32):
    ax2.text(xi, y + 1.5, f"{y:.0f}%", ha="center", fontsize=6)

ax2.set_xticks(list(x) + [gap_x])
ax2.set_xticklabels([power_of_2_label(n) for n in shared_N] + [""])
ax2.set_xlabel("Sequence length N")
ax2.set_ylabel("% of Cellular benefit erased")
ax2.set_title("Ablation effect: 7B vs. 32B")
ax2.legend(loc="upper left", frameon=False, fontsize=6.5)

fig.suptitle("Causal head ablation, cyclic_shift (alphabet=2, k=2): does model scale change how load-bearing the sink mechanism is?", y=1.05, fontsize=9)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "causal_ablation.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
