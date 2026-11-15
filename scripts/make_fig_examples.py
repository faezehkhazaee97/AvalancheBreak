#!/usr/bin/env python3
"""Annotated examples: real generations of Qwen2.5-32B rendered character by character.

Three rows per example: the input as shown to the model, the expected output, and the model's
output, with each character in a box; markers drawn as bars; output characters colored by
correctness (index scoring). Top: running_xor without markers. Middle: the same task with a marker
every 4 characters, where the character after each marker is the one that fails. Bottom: a
position-local digit task with markers, where every position is right.

Output: figs/fig_examples.pdf
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, INK, GRAY  # noqa: E402
from analyze_per_position import load  # noqa: E402
from analyze_aligned_scoring import index_vector, clean  # noqa: E402

TAG = "Qwen2.5-32B-Instruct"
N, K = 32, 4
OK, BAD, BAR = "#cfe7d6", "#f3c3b0", "#0072B2"


def pick(task, k, want_post_errors, axis=None):
    """A trial whose output has errors mostly at the post-marker positions (or, for k = 0, is mostly right)."""
    d = load(task, TAG)
    if axis is not None:
        d = d[d.Alphabet_Size.astype(int) == axis]
    x = d[(d.Seq_Length.astype(int) == N) & (d.Interval.astype(int) == k) & (d.Method == ("Base" if k == 0 else "Cellular_Bounds"))]
    best, score = None, -1
    for _, r in x.iterrows():
        t, p = clean(r.Ground_Truth), clean(r.LLM_Output)
        if len(t) != N or len(p) < N:
            continue
        v = index_vector(t, p)
        post = [j for j in range(1, N) if k and j % k == 0]
        if want_post_errors:
            s = (len(post) - v[post].sum()) * 3 + (v.sum() - (len(post) - v[post].sum()))   # many post errors, few others
        else:
            s = v.sum()
        if s > score:
            best, score = r, s
    return best


def draw(ax, shown, truth, pred, k, title):
    """Three rows of character boxes; bars drawn between boxes at the marker positions."""
    t, p = clean(truth), clean(pred)
    v = index_vector(t, p)
    rows = [("Input", clean(shown), None), ("Expected", t, None), ("Model output", p[:N].ljust(N), v)]
    w, h, gap = 1.0, 0.8, 0.28
    xs = []
    x = 0.0
    for j in range(N):
        if k and j and j % k == 0:
            x += gap
        xs.append(x); x += w
    for r, (name, s, corr) in enumerate(rows):
        y = -r * 1.05
        for j, ch in enumerate(s[:N]):
            face = "white" if corr is None else (OK if corr[j] else BAD)
            ax.add_patch(Rectangle((xs[j], y), w, h, facecolor=face, edgecolor="#bbbbbb", linewidth=0.4))
            ax.text(xs[j] + w / 2, y + h / 2, ch, ha="center", va="center", fontsize=6.5, family="cmtt10", color=INK)
        ax.text(-0.4, y + h / 2, name, ha="right", va="center", fontsize=6.5, color=INK)
    if k:
        for m in range(k, N, k):
            xb = xs[m] - gap / 2
            ax.plot([xb, xb], [-2 * 1.05 - 0.05, h + 0.05], color=BAR, linewidth=1.4)
            ax.annotate("", xy=(xs[m] + w / 2, -2 * 1.05 - 0.1), xytext=(xs[m] + w / 2, -2 * 1.05 - 0.45),
                        arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.6))
    ax.set_xlim(-4.2, x + 0.2); ax.set_ylim(-2 * 1.05 - 0.55, h + 0.45)
    ax.set_title(title, fontsize=8.5); ax.axis("off")


def main():
    set_rc()
    fig, axes = plt.subplots(3, 1, figsize=(MULTI_COL_WIDTH, 3.9))
    r0 = pick("running_xor", 0, False)
    draw(axes[0], r0.Input_Given, r0.Ground_Truth, r0.LLM_Output, 0, "(a) Running XOR, no markers")
    r1 = pick("running_xor", K, True)
    draw(axes[1], r1.Input_Given, r1.Ground_Truth, r1.LLM_Output, K, "(b) Running XOR, a marker every 4 characters (arrows: the position right after a marker)")
    r2 = pick("random_permutation_digits", K, False, axis=8)
    draw(axes[2], r2.Input_Given, r2.Ground_Truth, r2.LLM_Output, K, "(c) Random permutation of eight digits, a marker every 4 characters")
    fig.tight_layout(h_pad=0.6)
    savefig(fig, REPO / "figs" / "fig_examples.pdf")


if __name__ == "__main__":
    main()
