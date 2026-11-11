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


def _plain_labels(fig):
    """Put the paper's wording into any displayed text (titles, axis labels, legends); data
    filtering keeps using the internal names. Also drops figure-level titles: in the paper the
    caption carries the message, and several old titles used internal working names."""
    import re
    from matplotlib.text import Text
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    swaps = [("Cellular_Bounds", "Markers"), ("Cellular Bounds", "Markers"), ("Cellular", "Markers"),
             ("Token accuracy", "Character accuracy"), ("token accuracy", "character accuracy"),
             ("boundary-token", "marker"), ("Boundary-token", "Marker"),
             ("boundaries", "markers"), ("Boundaries", "Markers"),
             ("boundary", "marker"), ("Boundary", "Marker"),
             (" (no workaround)", ""), ("(space-sep.)", "(spaced)"), (" -- ", ", ")]
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s:
            continue
        new = re.sub(r"\bTAR\b", "Character accuracy", s)
        for a, b in swaps:
            new = new.replace(a, b)
        if "Bounds" not in new:  # keep "Bounds - Base", the paper's name for the difference
            new = re.sub(r"\bBase\b", "No markers", new)
        if new != s:
            t.set_text(new)


def savefig(fig, path):
    _plain_labels(fig)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Wrote {path}")


def power_of_2_label(value):
    """Plain tick label for a power of two: 64 -> '64', 0.125 -> '1/8'. Raises if the value is
    not a power of two (the study's grids are powers of two; k = 3 is excluded upstream), so a
    stray value surfaces as an error instead of a silently wrong label."""
    import math
    n = round(math.log2(value))
    if not math.isclose(value, 2 ** n, rel_tol=1e-6):
        raise ValueError(f"{value} is not a power of 2 -- exclude it before labeling, don't mislabel it")
    return str(2 ** n) if n >= 0 else f"1/{2 ** -n}"


def set_power_of_2_xticks(ax, values, fontsize=6.5, rotation=0):
    """Set explicit x-ticks at exactly the given (power-of-2) values, log2
    scale, labeled as 2^n. Use in place of matplotlib's default locator,
    which neither restricts itself to real data points nor formats them
    as powers of 2."""
    values = sorted(set(values))
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([power_of_2_label(v) for v in values], fontsize=fontsize, rotation=rotation)
    ax.minorticks_off()
