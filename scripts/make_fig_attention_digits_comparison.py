#!/usr/bin/env python3
"""Comparison figure: letter-alphabet (space-separated workaround) vs.
digit-alphabet (no workaround needed, matches real prompt format exactly)
attention capture. Answers directly: does the attention-sink mechanism
look any different once the tokenization confound is fully removed?

Panel (a): boundary-attention-mass-vs-density slopes, letter vs digit, by
length. Panel (b): entropy by layer, digit alphabet (Base vs Cellular).
Panel (c)-(d): same as the letter head-heatmap figure but for the digit
representative example, for the persistent 3-head core.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, power_of_2_label

set_rc()

RESULTS_DIR = REPO_ROOT / "results" / "Qwen2.5-7B-Instruct"
letters_csv = RESULTS_DIR / "attention_metrics_templated.csv"
digits_csv = RESULTS_DIR / "attention_metrics_digits.csv"
digits_npz = RESULTS_DIR / "attention_run_0_digits.npz"
print(f"Provenance: {letters_csv.relative_to(REPO_ROOT)}, {digits_csv.relative_to(REPO_ROOT)}, "
      f"{digits_npz.relative_to(REPO_ROOT)}")

letters = pd.read_csv(letters_csv)
digits = pd.read_csv(digits_csv)
npz = np.load(digits_npz, allow_pickle=True)
attn_base, attn_cell = npz["attn_base"], npz["attn_cell"]
tokens_base = [str(t) for t in npz["tokens_base"]]
tokens_cell = [str(t) for t in npz["tokens_cell"]]

fig = plt.figure(figsize=(MULTI_COL_WIDTH, 7.0))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

# (a) boundary-attention slope, letter vs digit, by length
ax_a = fig.add_subplot(gs[0, 0])
from scipy.stats import linregress
lengths = sorted(letters.Seq_Length.unique())
letter_slopes, digit_slopes = [], []
for N in lengths:
    for df, sink in [(letters, letter_slopes), (digits, digit_slopes)]:
        sub = df[df.Seq_Length == N]
        if sub.Inv_K.nunique() >= 2:
            slope, *_ = linregress(sub.Inv_K, sub.Cell_Boundary_Attn)
            sink.append(slope)
        else:
            sink.append(np.nan)
x = np.arange(len(lengths))
ax_a.bar(x - 0.18, letter_slopes, width=0.36, color="#7570b3", label="letters (space-sep.)")
ax_a.bar(x + 0.18, digit_slopes, width=0.36, color="#1b9e77", label="digits (no workaround)")
ax_a.set_xticks(x)
ax_a.set_xticklabels([power_of_2_label(n) for n in lengths], fontsize=6)
ax_a.set_xlabel("Sequence length N")
ax_a.set_ylabel("boundary-attn vs. density slope")
ax_a.set_title("(a) Sink strength: letters vs. digits")
ax_a.legend(loc="lower right", frameon=False, fontsize=6)

# (b) entropy by layer, digit alphabet
ax_b = fig.add_subplot(gs[0, 1])
eps = 1e-12
ent_base = (-np.sum(attn_base * np.log2(attn_base + eps), axis=-1)).mean(axis=(1, 2))
ent_cell = (-np.sum(attn_cell * np.log2(attn_cell + eps), axis=-1)).mean(axis=(1, 2))
layers = np.arange(len(ent_base))
ax_b.plot(layers, ent_base, color="black", linestyle="--", marker="o", markersize=3, label="Base")
ax_b.plot(layers, ent_cell, color="#1b9e77", linestyle="-", marker="s", markersize=3, label="Cellular_Bounds")
ax_b.set_xlabel("layer")
ax_b.set_ylabel("mean attention entropy (bits)")
ax_b.set_title("(b) Entropy by layer (digits)")
ax_b.legend(loc="best", frameon=False, fontsize=6)

# (c)-(d) head heatmaps, digit alphabet, persistent 3-head core (compact: show just 1 head as a spot-check)
HEADS = [(1, 24), (15, 5)]
for i, (layer, head) in enumerate(HEADS):
    for j, (attn, tokens, label) in enumerate([
        (attn_base, tokens_base, "Base"), (attn_cell, tokens_cell, "Cellular_Bounds"),
    ]):
        ax = fig.add_subplot(gs[1 + i, j])
        m = attn[layer, head]
        ax.imshow(m, cmap="viridis", vmin=0, vmax=np.percentile(m, 99), aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=4)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=4)
        for k, t in enumerate(tokens):
            if "|" in t:
                ax.axvline(k, color="red", alpha=0.25, linewidth=0.8)
                ax.axhline(k, color="red", alpha=0.25, linewidth=0.8)
        ax.set_title(f"L{layer}/H{head} -- {label} (digits)", fontsize=6.5)

fig.suptitle("Digit-alphabet attention capture (no tokenization confound, no space-separation workaround) "
              "vs. letter-alphabet comparison", y=1.0, fontsize=8)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "attention_digits_comparison.pdf"
savefig(fig, OUT)
