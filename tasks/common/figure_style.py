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

# Okabe-Ito palette (colorblind-safe), one fixed meaning per hue across every figure
OI = dict(blue="#0072B2", vermilion="#D55E00", green="#009E73", orange="#E69F00", purple="#CC79A7",
          sky="#56B4E9", yellow="#F0E442", black="#000000")
FORMAT_COLORS = {"letters": OI["blue"], "spaced": OI["vermilion"], "digits": OI["green"]}
AXIS_COLORS = {2: OI["blue"], 4: OI["orange"], 8: OI["purple"]}
INK, GRAY, GRID = "#1a1a1a", "#6e6e6e", "#e6e6e6"
BASE_STYLE = dict(color=INK, linestyle="--", linewidth=1.0, marker="o", markersize=2.6)
BOUNDS_STYLE = dict(linestyle="-", linewidth=1.2, marker="s", markersize=2.6)

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
    """House style: Computer Modern Roman (the paper's body font is Latin Modern), 8 pt, thin marks,
    light horizontal grid, no top or right spines, frameless legends, real text in the PDF."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "figure.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.title_fontsize": 7,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
        "lines.linewidth": 1.1, "lines.markersize": 2.8, "errorbar.capsize": 1.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titleweight": "normal", "axes.titlepad": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
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
             (" (no workaround)", ""), ("(space-sep.)", "(spaced)"), (" -- ", ", "),
             # glyphs absent from Computer Modern: use mathtext or plain equivalents
             ("\u2212", "-"), ("\u2014", "-"), ("\u2013", "-"), ("\u2190", r"$\leftarrow$"), ("\u2192", r"$\rightarrow$"),
             ("\u2265", r"$\geq$"), ("\u2264", r"$\leq$"), ("\u00b1", r"$\pm$"), ("\u2295", r"$\oplus$")]
    from matplotlib.ticker import FixedFormatter, FuncFormatter

    def fix(s):
        if not s:
            return s
        new = re.sub(r"\bTAR\b", "Character accuracy", s)
        for a, b in swaps:
            new = new.replace(a, b)
        if "Bounds" not in new:  # keep "Bounds - Base", the paper's name for the difference
            new = re.sub(r"\bBase\b", "No markers", new)
        return cm_safe_text(new)

    # Tick labels set with set_[xy]ticklabels live in a FixedFormatter (or, in recent matplotlib,
    # a FuncFormatter over the label list) and are re-rendered from it at draw time, so rewriting
    # the Text objects alone would not stick: rewrite the source.
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for fmt in (axis.get_major_formatter(), axis.get_minor_formatter()):
                if isinstance(fmt, FixedFormatter):
                    fmt.seq = [fix(str(s)) for s in fmt.seq]
                elif isinstance(fmt, FuncFormatter) and not getattr(fmt.func, "_plain_labels", False):
                    def wrapped(x, pos=None, _f=fmt.func):
                        return fix(str(_f(x, pos)))
                    wrapped._plain_labels = True
                    fmt.func = wrapped
    for t in fig.findobj(Text):
        s = t.get_text()
        new = fix(s)
        if new != s:
            t.set_text(new)


def cm_safe_text(s):
    """Rewrite the ASCII characters that Computer Modern Roman (cmr10) has no glyph for, outside
    any $...$ math: in cmr10's slots, '_' prints as a raised dot accent, '|' as an em dash, and
    '<' / '>' as inverted exclamation and question marks. Task names (cyclic_shift, ...) become
    typewriter text with a real underscore, as \\task{} sets them in the paper; a bare '|' becomes
    a math bar. Each line is handled on its own (matplotlib parses math per line), and a line
    with an unbalanced '$' is left alone."""
    import re
    out = []
    for line in s.split("\n"):
        if line.replace(r"\$", "").count("$") % 2:
            out.append(line)
            continue
        parts = re.split(r"(?<!\\)(\$[^$]*(?<!\\)\$)", line)
        for i, p in enumerate(parts):
            if p.startswith("$") or not p:
                continue
            p = re.sub(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]*)+|_+[A-Za-z0-9]+(?:_[A-Za-z0-9]*)*",
                       lambda m: r"$\mathtt{" + m.group(0).replace("_", r"\_") + "}$", p)
            sub = re.split(r"(\$[^$]*\$)", p)   # leave the \mathtt pieces just made alone
            for j, q in enumerate(sub):
                if not q.startswith("$"):
                    sub[j] = (q.replace("_", r"$\_$").replace("|", "$|$")
                              .replace("<", "$<$").replace(">", "$>$"))
            parts[i] = "".join(sub)
        out.append("".join(parts))
    return "\n".join(out)


def align_panel_titles(fig):
    """Start every lettered panel title ("(a) ...") at the panel's leftmost content -- its y tick
    labels and y-axis label included -- rather than at the edge of the plotting area, so the letter
    sits at the true top-left corner of each panel."""
    import re as _re
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        t = ax._left_title
        if not _re.match(r"\([a-z]\)", t.get_text() or ""):
            continue
        t.set_visible(False)
        bb = ax.get_tightbbox(renderer)
        t.set_visible(True)
        box = ax.get_window_extent(renderer)
        if bb is None or box.width == 0:
            continue
        t.set_x((bb.x0 - box.x0) / box.width)


def savefig(fig, path):
    _plain_labels(fig)
    align_panel_titles(fig)
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
