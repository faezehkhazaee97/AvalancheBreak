#!/usr/bin/env python3
"""Figure: real letter tokenization in the attention map (figs/
attention_real_letters.pdf). Shows the persistent 3-head core's attention
on the ACTUAL prompt format used by the accuracy grid -- concatenated
letters, no space-separation workaround -- so the token-merging confound
is directly visible in the heatmap axes themselves, not just inferred from
audit statistics. Base collapses 16 characters into 6 tokens (heavy
merging); Cellular_Bounds recovers most (but not all) of the per-character
resolution -- 15 tokens for the same 16 characters + boundaries, since
within-chunk pairs (k=2) can still merge two characters into one token.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH

set_rc()

NPZ = REPO_ROOT / "results" / "Qwen2.5-7B-Instruct" / "attention_run_0_real_letters.npz"
print(f"Provenance: {NPZ.relative_to(REPO_ROOT)}")
npz = np.load(NPZ, allow_pickle=True)
attn_base, attn_cell = npz["attn_base"], npz["attn_cell"]
tokens_base = [str(t) for t in npz["tokens_base"]]
tokens_cell = [str(t) for t in npz["tokens_cell"]]
raw_chars = str(npz["raw_chars"])

print(f"Raw characters ({len(raw_chars)}): {raw_chars}")
print(f"Base tokens ({len(tokens_base)}): {tokens_base}")
print(f"Cellular tokens ({len(tokens_cell)}): {tokens_cell}")

HEADS = [(1, 24), (3, 24), (15, 5)]

fig, axes = plt.subplots(len(HEADS), 2, figsize=(MULTI_COL_WIDTH, 2.6 * len(HEADS)))

for row, (layer, head) in enumerate(HEADS):
    for col, (attn, tokens, label) in enumerate([
        (attn_base, tokens_base, f"Base ({len(tokens_base)} tokens for {len(raw_chars)} chars -- MERGED)"),
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
                ax.axvline(i, color="red", alpha=0.3, linewidth=1)
                ax.axhline(i, color="red", alpha=0.3, linewidth=1)
        ax.set_title(f"L{layer}/H{head}\n{label}", fontsize=6.5)

fig.suptitle(f"Real (concatenated) letter tokenization: raw='{raw_chars}' -- "
              f"Base merges into {len(tokens_base)} tokens, Cellular_Bounds into {len(tokens_cell)}", y=1.01, fontsize=8)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "attention_real_letters.pdf"
savefig(fig, OUT)
