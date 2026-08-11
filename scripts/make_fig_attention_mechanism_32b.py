#!/usr/bin/env python3
"""Figure 5 rebuild: Attention mechanism panel, Qwen2.5-32B-Instruct, REAL
(concatenated) letter tokenization -- replaces the old Qwen-7B/space-
separated version. Same four panels: (a) boundary-attention mass vs.
density, faceted by length; (b) attention entropy by layer, Base vs.
Cellular; (c) off-diagonal attention mass distribution; (d) attention-
weighted distance-decay profile. Reuses the exact same sweep/capture data
already built for the main-paper Figure 1 (attn-32b-realletters job) --
no new GPU run needed, just a different panel layout of the same data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, set_power_of_2_xticks

set_rc()

RESULTS_DIR = REPO_ROOT / "results" / "Qwen2.5-32B-Instruct"
SWEEP_CSV = RESULTS_DIR / "attention_metrics_32b_realletters.csv"
NPZ = RESULTS_DIR / "attention_run_0_32b_realletters.npz"
print(f"Provenance -- rebuilt Figure 5 sources: {SWEEP_CSV.relative_to(REPO_ROOT)}, {NPZ.relative_to(REPO_ROOT)}")

sweep = pd.read_csv(SWEEP_CSV)
sweep = sweep[sweep.Alphabet_Size == 2] if "Alphabet_Size" in sweep.columns else sweep
npz = np.load(NPZ, allow_pickle=True)
attn_base, attn_cell = npz["attn_base"], npz["attn_cell"]  # (layers, heads, seq, seq)
tokens_cell = npz["tokens_cell"]

fig, axes = plt.subplots(2, 2, figsize=(MULTI_COL_WIDTH, 6.0))

# (a) boundary-attention mass vs density, faceted by length
ax = axes[0, 0]
sweep_a = sweep[sweep.Interval != 3]  # k=3 excluded per this study's own pooled-analysis convention
for N, sub in sweep_a.groupby("Seq_Length"):
    agg = sub.groupby("Inv_K")["Cell_Boundary_Attn"].mean()
    ax.plot(agg.index, agg.values, marker="o", markersize=3, linewidth=1.2, label=f"N={N}")
ax.set_xlabel("density (1/k)")
ax.set_ylabel("boundary-token attention mass")
ax.set_title("(a) Boundary attention vs. density")
set_power_of_2_xticks(ax, sweep_a.Inv_K.unique(), fontsize=6)
ax.legend(loc="upper left", frameon=False, fontsize=5.5, ncol=2)

# (b) entropy by layer, Base vs Cellular (representative example)
ax = axes[0, 1]
eps = 1e-12
ent_base = (-np.sum(attn_base * np.log2(attn_base + eps), axis=-1)).mean(axis=(1, 2))  # per layer
ent_cell = (-np.sum(attn_cell * np.log2(attn_cell + eps), axis=-1)).mean(axis=(1, 2))
layers = np.arange(len(ent_base))
ax.plot(layers, ent_base, color="black", linestyle="--", marker="o", markersize=3, label="Base")
ax.plot(layers, ent_cell, color="#eb6834", linestyle="-", marker="s", markersize=3, label="Cellular_Bounds")
ax.set_xlabel("layer")
ax.set_ylabel("mean attention entropy (bits)")
ax.set_title("(b) Entropy by layer")
ax.legend(loc="best", frameon=False, fontsize=6)

# (c) off-diagonal attention mass distribution
ax = axes[1, 0]
def offdiag_values(t):
    n = t.shape[-1]
    mask = ~np.eye(n, dtype=bool)
    return t[:, :, mask].ravel()
od_base = offdiag_values(attn_base)
od_cell = offdiag_values(attn_cell)
ax.hist(od_base, bins=60, range=(0, 0.3), density=True, histtype="step", color="black",
        linestyle="--", linewidth=1.2, label="Base")
ax.hist(od_cell, bins=60, range=(0, 0.3), density=True, histtype="step", color="#eb6834",
        linewidth=1.2, label="Cellular_Bounds")
ax.set_xlabel("off-diagonal attention probability")
ax.set_ylabel("density")
ax.set_title("(c) Off-diagonal attention distribution")
ax.legend(loc="upper right", frameon=False, fontsize=6)

# (d) attention-weighted distance-decay profile
ax = axes[1, 1]
def distance_profile(t):
    n = t.shape[-1]
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    mean_attn = t.mean(axis=(0, 1))  # (seq, seq), averaged over layers/heads
    profile = np.zeros(n)
    for d in range(n):
        mask = dist == d
        profile[d] = mean_attn[mask].mean() if mask.any() else np.nan
    return profile

prof_base = distance_profile(attn_base)
prof_cell = distance_profile(attn_cell)
ax.plot(np.arange(len(prof_base)), prof_base, color="black", linestyle="--", marker="o", markersize=3, label="Base")
ax.plot(np.arange(len(prof_cell)), prof_cell, color="#eb6834", linestyle="-", marker="s", markersize=3, label="Cellular_Bounds")
ax.set_yscale("log")
ax.set_xlabel("|query - key| distance")
ax.set_ylabel("mean attention probability")
ax.set_title("(d) Distance-decay profile")
ax.legend(loc="upper right", frameon=False, fontsize=6)

fig.suptitle("Qwen2.5-32B attention mechanism (cyclic_shift, alphabet=2, real tokenization)", y=1.01)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "attention_mechanism_32b.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
