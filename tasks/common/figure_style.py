"""Shared plotting style for the arXiv-manuscript figures (figs/*.pdf).
Base = black dashed; Cellular_Bounds = colored solid, one color per axis
value (alphabet/modulus), consistent across figures. Vector PDF output,
sized for a two-column layout: SINGLE_COL ~3.3in wide, MULTI_COL ~7in wide.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINGLE_COL_WIDTH = 3.3
MULTI_COL_WIDTH = 7.0

AXIS_COLORS = {
    2: "#1b9e77", 4: "#d95f02", 8: "#7570b3",
}
BASE_STYLE = dict(color="black", linestyle="--", linewidth=1.3, marker="o", markersize=3)
BOUNDS_STYLE = dict(linestyle="-", linewidth=1.6, marker="s", markersize=3)

MODEL_ORDER = [
    "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct",
    "Mistral-7B-Instruct-v0.3", "Phi-3.5-mini-instruct",
]
MODEL_LABELS = {
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "Qwen2.5-32B-Instruct": "Qwen2.5-32B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Phi-3.5-mini-instruct": "Phi-3.5-mini",
}


def set_rc():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed as real text, not paths
        "axes.spines.top": False, "axes.spines.right": False,
    })


def savefig(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Wrote {path}")
