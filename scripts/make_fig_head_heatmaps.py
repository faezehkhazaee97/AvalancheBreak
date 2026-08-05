#!/usr/bin/env python3
"""Figure 6: Attention head heatmaps (figs/head_heatmaps.pdf). Layer/head
attention heatmaps, Base vs. Cellular (boundaries visible), for the
persistent 3-head core identified in the causal ablation (layer 1/head 24,
layer 3/head 24, layer 15/head 5) plus the strongest single boundary head
from the diagnostic ranking, using the representative captured example
(N=16, k=2, alphabet=2). Replaces old Figures 12-14.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH

set_rc()

NPZ = REPO_ROOT / "results" / "Qwen2.5-7B-Instruct" / "attention_run_0_templated.npz"
print(f"Provenance -- Figure 6 source: {NPZ.relative_to(REPO_ROOT)}")
npz = np.load(NPZ, allow_pickle=True)
attn_base, attn_cell = npz["attn_base"], npz["attn_cell"]
tokens_base = [str(t) for t in npz["tokens_base"]]
tokens_cell = [str(t) for t in npz["tokens_cell"]]

HEADS = [(1, 24), (3, 24), (15, 5)]  # the persistent 3-head core (Section 4.3)

fig, axes = plt.subplots(len(HEADS), 2, figsize=(MULTI_COL_WIDTH, 2.3 * len(HEADS)))

for row, (layer, head) in enumerate(HEADS):
    for col, (attn, tokens, label) in enumerate([
        (attn_base, tokens_base, "Base"), (attn_cell, tokens_cell, "Cellular_Bounds"),
    ]):
        ax = axes[row, col]
        m = attn[layer, head]
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=np.percentile(m, 99), aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=4)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=4)
        # mark boundary token positions with a red tick
        for i, t in enumerate(tokens):
            if "|" in t:
                ax.axvline(i, color="red", alpha=0.25, linewidth=0.8)
                ax.axhline(i, color="red", alpha=0.25, linewidth=0.8)
        ax.set_title(f"L{layer}/H{head} -- {label}", fontsize=7)
        if col == 0:
            ax.set_ylabel("query", fontsize=6)
        if row == len(HEADS) - 1:
            ax.set_xlabel("key", fontsize=6)

fig.suptitle("Persistent 3-head boundary-attention core: Base vs. Cellular_Bounds "
              "(red lines = boundary token positions)", y=1.0, fontsize=8)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "head_heatmaps.pdf"
savefig(fig, OUT)
