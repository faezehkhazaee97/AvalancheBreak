#!/usr/bin/env python3
"""Per-cell scatter of the two per-position effects. Each point is one (task, model) cell: the change
with markers at the within-segment positions (x) against the change at the position right after a
marker (y), each relative to the no-marker run at the same positions (N <= 64, k in {2,4,8,16}).
Position-local cells sit near the diagonal (both positions gain alike or the post-marker position
gains more); dependency cells sit below the axis (within gains, post-marker loses).

Output: figs/fig_cell_scatter.pdf
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.figure_style import set_rc, savefig, SINGLE_COL_WIDTH, INK, GRAY  # noqa: E402

BLOCKS = [("Position-local, letters", "Position-local, letters", "#56B4E9", "o"),
          ("Position-local, spaced letters", "Position-local, spaced letters", "#0072B2", "s"),
          ("Position-local, digits", "Position-local, digits", "#009E73", "^"),
          ("Dependency, as is", "Dependency tasks", "#D55E00", "D"),
          ("Dependency severed (reset)", "Dependency tasks, reset variant", "#CC79A7", "v")]


def main():
    set_rc()
    v = pd.read_csv(REPO / "results_clean" / "_summary" / "per_position_values.csv")
    v["dx"] = v.within - v.base_at_within; v["dy"] = v.post - v.base_at_post
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH + 0.4, 3.0))
    lim = 0.62
    ax.axhline(0, color=GRAY, linewidth=0.6); ax.axvline(0, color=GRAY, linewidth=0.6)
    ax.plot([-lim, lim], [-lim, lim], color=GRAY, linewidth=0.6, linestyle=":")
    for block, lab, col, mk in BLOCKS:
        x = v[v.block == block]
        if x.empty:
            continue
        ax.scatter(x.dx, x.dy, s=16, color=col, marker=mk, label=f"{lab} ({len(x)})", alpha=0.85, linewidths=0.3, edgecolors="white", zorder=3)
    ax.set_xlim(-0.3, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Change within a segment"); ax.set_ylabel("Change right after a marker")
    ax.grid(True, axis="both", color="#e6e6e6", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=6, frameon=False, handletextpad=0.3)
    ax.text(lim - 0.02, -lim + 0.03, "below the axis: the blind spot", ha="right", va="bottom", fontsize=6, color=GRAY)
    fig.tight_layout()
    savefig(fig, REPO / "figs" / "fig_cell_scatter.pdf")


if __name__ == "__main__":
    main()
