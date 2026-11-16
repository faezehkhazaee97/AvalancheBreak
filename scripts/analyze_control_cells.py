#!/usr/bin/env python3
"""Every pooled cell of the two tokenization controls (spaced letters, digits), task by model.

Per cell (N <= 64, k in {1, 2, 4, 8, 16}, all alphabet sizes, k = 3 excluded): the length-matched
no-marker accuracy, the marker accuracy, Delta, the share of remaining errors removed
Delta / (1 - Base_matched), and two z values:
  z          the primary analytic z. Its standard error treats the (N, k, A) cells as fixed
             design strata on the marker side and the alphabet sizes within each N on the
             baseline side, SE(Bounds)^2 = sum_c w_c^2 s_c^2 / n_c with w_c the share of marker
             generations in cell c (tasks/common/pooled.py). Accuracy varies steeply across
             those dimensions, and the grid is fixed rather than sampled, so the unstratified
             s/sqrt(n) over all of a cell's marker generations would charge that deterministic
             between-stratum variation to sampling noise. The unstratified value is still
             computed, as z_unstrat in the CSV, but is no longer printed in the table;
  z_cluster  the cell-level bootstrap of Appendix C (statistics_robustness.csv), open models
             only, which remains the conservative check.
The same two standard errors are also computed at N = 64 alone (z64, z64_unstrat), where the
strata are (k, A).

Model-level rows. Equation 3 defines rho at the model level, as a ratio of means: the unweighted
mean Delta over a model's three tasks divided by one minus the mean length-matched no-marker
accuracy. The per-task "share removed" column is Delta / (1 - no-marker) for one task and does
not average to it, so the table also carries one summary row per model and format with the
three-task mean baseline, mean marker accuracy, mean Delta, mean Delta at N = 64, and the
model-level rho, in a block of its own below the per-task rows.

Also printed: per model and format, the mean over the two strictly isomorphic tasks (cyclic_shift,
random_permutation) next to conditional_shift, whose digit version is easier by construction.
Outputs: results_clean/_summary/control_cells.csv, paper_tmlr/tables/control_cells.tex
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS  # noqa: E402
from tasks.common.pooled import stratified_var  # noqa: E402

LENGTHS = [4, 8, 16, 32, 64]
SIG_Z = 3.54
TASKS = ["cyclic_shift", "random_permutation", "conditional_shift"]
FORMATS = [("_spaced", "Spaced"), ("_digits", "Digits")]
STRONG = ["Qwen2.5 14B", "Qwen2.5 32B", "Qwen3.8 27B", "Gemma 4 31B"]
SMALL = ["Qwen2.5 7B", "Mistral 7B", "Phi-3.5 mini"]


def strat(d):
    d = d[d.Interval.astype(int) != 3].copy()
    d["N"] = d.Seq_Length.astype(int); d["k"] = d.Interval.astype(int); d["TAR"] = d.TAR.astype(float)
    a = next((c for c in ("Alphabet_Size", "Modulus", "Axis_Val") if c in d.columns), None)
    d["A"] = d[a].astype(str) if a else "0"
    d = d[d.N.isin(LENGTHS)]
    b, c = d[d.k == 0], d[d.k > 0]
    p = c.groupby("N").size() / len(c)
    base_N = b.groupby("N").TAR.mean().reindex(p.index)
    base_m = float((p * base_N).sum()); bounds_m = float(c.TAR.mean()); delta = bounds_m - base_m
    se_b_N = b.groupby("N").TAR.sem().reindex(p.index).fillna(0)
    se_u = float(np.sqrt(c.TAR.sem() ** 2 + (p ** 2 * se_b_N ** 2).sum()))
    # primary: marker side over the fixed (N, k, A) cells, baseline side over A within each N
    var_c, thin = stratified_var(c, ["N", "k", "A"])
    var_b = 0.0
    for N, pN in p.items():
        v, t = stratified_var(b[b.N == N], ["A"])
        var_b += float(pN) ** 2 * v; thin += t
    se = float(np.sqrt(var_c + var_b))
    c64, b64 = c[c.N == 64], b[b.N == 64]
    d64 = z64 = z64_u = np.nan
    if len(c64) and len(b64):
        d64 = float(c64.TAR.mean() - b64.TAR.mean())
        v64, _ = stratified_var(c64, ["k", "A"])
        vb64, _ = stratified_var(b64, ["A"])
        se64 = float(np.sqrt(v64 + vb64))
        se64_u = float(np.sqrt(c64.TAR.sem() ** 2 + b64.TAR.sem() ** 2))
        z64 = d64 / se64 if se64 > 0 else np.nan
        z64_u = d64 / se64_u if se64_u > 0 else np.nan
    return dict(base=base_m, bounds=bounds_m, delta=delta, se=se, z=delta / se,
                se_unstrat=se_u, z_unstrat=delta / se_u, thin_strata=thin,
                share=delta / (1 - base_m) if base_m < 0.98 else np.nan,
                delta64=d64, z64=z64, z64_unstrat=z64_u)


def model_level(g):
    """Equation 3 at the model level: mean Delta over the model's tasks, over one minus the
    mean length-matched no-marker accuracy. Dashed at ceiling, as in the per-task column."""
    base, delta = g.base.mean(), g.delta.mean()
    rho = delta / (1 - base) if base < 0.98 else np.nan
    return dict(base=base, bounds=g.bounds.mean(), delta=delta, rho=rho, delta64=g.delta64.mean())


def report(t):
    """Everything the prose quotes, old (unstratified) against new (stratified)."""
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    if t.thin_strata.sum():
        print(f"\nWARNING: {int(t.thin_strata.sum())} strata hold a single generation "
              "(given the pooled within-stratum variance)")

    def counts(sub):
        return (int((sub.z_unstrat.abs() >= SIG_Z).sum()), int((sub.z.abs() >= SIG_Z).sum()), len(sub))

    print("\n== significance counts, |z| >= 3.54: unstratified (old) -> stratified (new) ==")
    groups = [("all 48 control cells", t),
              ("4 stronger models, both formats (paper: 23 of 24)", t[t.model.isin(STRONG)]),
              ("4 stronger models, spaced (paper: all 12)", t[t.model.isin(STRONG) & (t.format == "Spaced")]),
              ("4 stronger models, digits (paper: 11 of 12)", t[t.model.isin(STRONG) & (t.format == "Digits")]),
              ("3 smallest models, both formats (paper: 7 of 18)", t[t.model.isin(SMALL)]),
              ("Gemini 3.6 Flash (paper: none significant)", t[t.model == "Gemini 3.6 Flash"])]
    for name, sub in groups:
        if not len(sub):
            continue
        old, new, n = counts(sub)
        print(f"  {name}: {old} of {n} -> {new} of {n}")
        for r in sub.itertuples():
            was, now = abs(r.z_unstrat) >= SIG_Z, abs(r.z) >= SIG_Z
            if was != now:
                print(f"      FLIPS {'to significant' if now else 'to NOT significant'}: "
                      f"{r.model} {r.format} {r.task}  z {r.z_unstrat:+.2f} -> {r.z:+.2f}")
    larger = int((t.z.abs() > t.z_unstrat.abs()).sum())
    print(f"  stratified |z| larger in {larger} of {len(t)} cells (paper: 44 of 48)")

    print("\n== at N = 64 alone (paper: all 24 cells at the 4 stronger models clear 3.54) ==")
    s64 = t[t.model.isin(STRONG)].dropna(subset=["z64"])
    print(f"  {int((s64.z64_unstrat.abs() >= SIG_Z).sum())} of {len(s64)} -> "
          f"{int((s64.z64.abs() >= SIG_Z).sum())} of {len(s64)}; "
          f"delta range {s64.delta64.min():+.2f} to {s64.delta64.max():+.2f} (mean {s64.delta64.mean():+.2f}); "
          f"min |z64| {s64.z64_unstrat.abs().min():.2f} -> {s64.z64.abs().min():.2f}")

    print("\n== model-level rho of Equation 3, all 8 models x 2 formats ==")
    print(f"  {'Format':8s} {'Model':17s} {'mean base':>9s} {'mean bounds':>11s} {'mean delta':>10s} "
          f"{'rho':>7s} {'rho (iso 2 tasks)':>18s} {'mean d64':>9s}")
    for (fmt, model), g in t.groupby(["format", "model"], sort=False):
        m = model_level(g)
        iso = model_level(g[g.task != "conditional_shift"])
        print(f"  {fmt:8s} {model:17s} {m['base']:9.4f} {m['bounds']:11.4f} {m['delta']:+10.4f} "
              f"{m['rho'] * 100 if np.isfinite(m['rho']) else float('nan'):6.2f}% "
              f"{iso['rho'] * 100 if np.isfinite(iso['rho']) else float('nan'):17.2f}% {m['delta64']:+9.4f}")

    print("\nmodel means: base, delta, share over all three tasks vs the two isomorphic tasks; conditional_shift alone")
    for (fmt, model), g in t.groupby(["format", "model"], sort=False):
        iso = g[g.task != "conditional_shift"]; cs = g[g.task == "conditional_shift"]
        sh = lambda x: x.delta.mean() / (1 - x.base.mean())
        print(f"  {fmt:7s} {model:16s} all: base {g.base.mean():.3f} delta {g.delta.mean():+.3f} share {sh(g):.0%} | "
              f"isomorphic: base {iso.base.mean():.3f} delta {iso.delta.mean():+.3f} share {sh(iso):.0%} | cond: base {cs.base.mean():.3f} delta {cs.delta.mean():+.3f}")


def main():
    rob = pd.read_csv(REPO / "results_clean" / "_summary" / "statistics_robustness.csv")
    rows = []
    for suf, fmt in FORMATS:
        for task in TASKS:
            for label, tag in MODELS:
                if tag is None:
                    continue
                d = load(task + suf, tag)
                if d is None:
                    continue
                r = strat(d)
                m = rob[(rob.task == task + suf) & (rob.model == label)]
                r.update(format=fmt, task=task, model=label, z_cluster=float(m.cluster_z.iloc[0]) if len(m) else np.nan)
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(REPO / "results_clean" / "_summary" / "control_cells.csv", index=False)
    report(t)

    names = {"cyclic_shift": "\\task{cyclic\\_shift}", "random_permutation": "\\task{random\\_permutation}",
             "conditional_shift": "\\task{conditional\\_shift}"}
    f = lambda v: "--" if pd.isna(v) else f"{v:.1f}"
    # Ten columns at \scriptsize have to fit \linewidth on their own (main.tex \input's this
    # table without a \resizebox), and the multi-word headings, not the data, set the width: they
    # are stacked onto two lines and \tabcolsep is reduced.
    hdr = lambda *ls: "\\begin{tabular}[b]{@{}c@{}}" + "\\\\".join(ls) + "\\end{tabular}"
    head = ["Model", "Format", "Task", hdr("No", "markers"), "Markers", "$\\Delta$",
            hdr("Share", "removed"), hdr("$z$", "(stratified)"),
            hdr("$z$ (cell", "bootstrap)"), hdr("$\\Delta$ at", "$N = 64$")]
    NCOL = len(head)
    lines = ["{\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{lll ccc c cc c}", "\\toprule",
             " & ".join(head) + " \\\\", "\\midrule"]
    d64 = lambda v: "--" if pd.isna(v) else f"${v:+.2f}$".replace("$-0.00$", "$0.00$").replace("$+0.00$", "$0.00$")

    def pct(v):
        """Share of remaining errors removed; blank at ceiling, where the denominator vanishes.
        A negative share gets a math minus, not an ASCII hyphen (tidy_tables.py only repairs bare
        hyphens in front of a decimal, so a whole-number percentage has to be written correctly
        here)."""
        if not np.isfinite(v):
            return "--"
        n = int(round(100 * v))
        return f"$-{abs(n)}\\%$" if n < 0 else f"{n}\\%"

    for model in t.model.unique():
        g = t[t.model == model]; first = True
        for r in g.itertuples():
            lines.append(f"{model if first else ''} & {r.format} & {names[r.task]} & {r.base:.2f} & {r.bounds:.2f} & ${r.delta:+.2f}$ & "
                         f"{pct(r.share)} & {r.z:.1f} & {f(r.z_cluster)} & {d64(r.delta64)} \\\\".replace("$-0.00$", "$0.00$").replace("$+0.00$", "$0.00$"))
            first = False
        lines.append("\\addlinespace")
    # model-level block: Equation 3 is a ratio of means over a model's three tasks, so it cannot
    # be read off the per-task share column; it gets rows of its own, with no z (a mean over
    # tasks has no single standard error) and no cell bootstrap.
    lines[-1] = "\\midrule"
    # No \Eqref here: tables/ is shared with the other build, so the block heading must not
    # depend on a macro defined in one main.tex. The caption names the equation.
    lines.append(f"\\multicolumn{{{NCOL}}}{{l}}{{\\emph{{Model level: mean over the model's three tasks, "
                 "and the model-level $\\rho$}} \\\\")
    for model in t.model.unique():
        first = True
        for fmt in ("Spaced", "Digits"):
            g = t[(t.model == model) & (t.format == fmt)]
            if g.empty:
                continue
            m = model_level(g)
            lines.append(f"{model if first else ''} & {fmt} & \\emph{{Three-task mean}} & {m['base']:.2f} & {m['bounds']:.2f} & "
                         f"${m['delta']:+.2f}$ & {pct(m['rho'])} & -- & -- & {d64(m['delta64'])} \\\\"
                         .replace("$-0.00$", "$0.00$").replace("$+0.00$", "$0.00$"))
            first = False
    lines += ["\\bottomrule", "\\end{tabular}}"]
    (REPO / "paper_tmlr" / "tables" / "control_cells.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/control_cells.tex")


if __name__ == "__main__":
    main()
