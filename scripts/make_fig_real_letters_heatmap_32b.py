#!/usr/bin/env python3
"""Figure rebuild: real letter tokenization in the attention map, Qwen2.5-
32B-Instruct (was Qwen-7B). Shows the persistent 3-head core's attention
on the ACTUAL prompt format used by the accuracy grid -- concatenated
letters, no space-separation workaround -- so the token-merging confound
is directly visible in the heatmap axes themselves. Reuses the same top-3
heads already re-identified for 32B in make_fig_head_heatmaps_32b.py
(L21/H1, L37/H1, L1/H7) rather than the old 7B indices (1,24)/(3,24)/(15,5),
which don't correspond to anything on 32B's different architecture.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH

set_rc()

NPZ = REPO_ROOT / "results" / "Qwen2.5-32B-Instruct" / "attention_run_0_32b_realletters.npz"
print(f"Provenance: {NPZ.relative_to(REPO_ROOT)}")
npz = np.load(NPZ, allow_pickle=True)
attn_base, attn_cell = npz["attn_base"], npz["attn_cell"]
tokens_base = [str(t) for t in npz["tokens_base"]]
tokens_cell = [str(t) for t in npz["tokens_cell"]]
n_raw_chars = int(npz["N_length"])

HEADS = [(21, 1), (37, 1), (1, 7)]  # same top-3 heads as make_fig_head_heatmaps_32b.py

fig, axes = plt.subplots(len(HEADS), 2, figsize=(MULTI_COL_WIDTH, 2.6 * len(HEADS)))

for row, (layer, head) in enumerate(HEADS):
    for col, (attn, tokens, label) in enumerate([
        (attn_base, tokens_base, f"Base ({len(tokens_base)} tokens for {n_raw_chars} raw chars -- MERGED)"),
        (attn_cell, tokens_cell, f"Cellular_Bounds ({len(tokens_cell)} tokens)"),
    ]):
        ax = axes[row, col]
        m = attn[layer, head]
        ax.imshow(m, cmap="viridis", vmin=0, vmax=np.percentile(m, 99), aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=6)
        for i, t in enumerate(tokens):
            if "|" in t:
                ax.axvline(i, color="#eb6834", alpha=0.4, linewidth=1)
                ax.axhline(i, color="#eb6834", alpha=0.4, linewidth=1)
        ax.set_title(f"L{layer}/H{head}\n{label}", fontsize=6.5)

fig.suptitle(f"Qwen2.5-32B-Instruct, real (concatenated) letter tokenization ({n_raw_chars} raw chars) -- "
              f"Base merges into {len(tokens_base)} tokens, Cellular_Bounds into {len(tokens_cell)}", y=1.01, fontsize=8)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "attention_real_letters_32b.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
