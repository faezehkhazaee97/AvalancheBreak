#!/usr/bin/env python3
"""Figure 6 rebuild: Persistent boundary-attention head core, Qwen2.5-32B-
Instruct, REAL (concatenated) letter tokenization. Replaces the old
Qwen-7B/space-separated version, whose hardcoded head indices (1,24),
(3,24), (15,5) are meaningless on 32B -- that model has 64 layers x 40
heads, a completely different index space from 7B's architecture. This
re-identifies the top-3 boundary-attending heads directly from the 32B
capture (same data Figure 1 uses) rather than reusing 7B indices.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH

set_rc()

RESULTS_DIR = REPO_ROOT / "results" / "Qwen2.5-32B-Instruct"
NPZ = RESULTS_DIR / "attention_run_0_32b_realletters.npz"
print(f"Provenance: {NPZ.relative_to(REPO_ROOT)}")

npz = np.load(NPZ, allow_pickle=True)
attn_base, attn_cell = npz["attn_base"], npz["attn_cell"]
tokens_base = [str(t) for t in npz["tokens_base"]]
tokens_cell = [str(t) for t in npz["tokens_cell"]]

n_layers, n_heads = attn_cell.shape[0], attn_cell.shape[1]
boundary_idx = [i for i, t in enumerate(tokens_cell) if "|" in t]

scores = np.zeros((n_layers, n_heads))
for l in range(n_layers):
    for h in range(n_heads):
        scores[l, h] = attn_cell[l, h][:, boundary_idx].sum() if boundary_idx else 0

top3_flat = np.argsort(scores.ravel())[::-1][:3]
TOP_HEADS = [tuple(np.unravel_index(idx, scores.shape)) for idx in top3_flat]
print(f"Top-3 boundary-attending heads on Qwen-32B: {TOP_HEADS}")

fig, axes = plt.subplots(len(TOP_HEADS), 2, figsize=(MULTI_COL_WIDTH, 2.6 * len(TOP_HEADS)))

for row, (layer, head) in enumerate(TOP_HEADS):
    for col, (attn, tokens, label) in enumerate([
        (attn_base, tokens_base, "Base"),
        (attn_cell, tokens_cell, "Cellular_Bounds"),
    ]):
        ax = axes[row, col]
        m = attn[layer, head]
        ax.imshow(m, cmap="viridis", vmin=0, vmax=np.percentile(m, 99), aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=5)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=5)
        for i, t in enumerate(tokens):
            if "|" in t:
                ax.axvline(i, color="#eb6834", alpha=0.4, linewidth=0.8)
                ax.axhline(i, color="#eb6834", alpha=0.4, linewidth=0.8)
        ax.set_title(f"L{layer}/H{head} -- {label}", fontsize=7)
        if col == 0:
            ax.set_ylabel("query", fontsize=6)
        ax.set_xlabel("key", fontsize=6)

fig.suptitle("Persistent 3-head boundary-attention core, Qwen2.5-32B-Instruct "
             "(Base vs. Cellular_Bounds; orange lines = boundary token positions)",
             y=1.01)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "head_heatmaps_32b.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
