#!/usr/bin/env python3
"""Analysis of scripts/run_marker_sides.py: is the benefit from segmenting the input, from
asking for a segmented output, or both?

Five conditions at Qwen2.5-32B (N in {32, 64}, k in {4, 16} pooled, 20 trials):
  base_clean  no markers either side, and no sentence about bars in the prompt
  base_instr  no markers either side, marker sentence present   [main grid Base]
  in_only     markers in the input, output asked for without them
  out_only    no markers in the input, output asked for with them
  in_out      markers in the input, copied to the output        [main grid Bounds]

base_instr and in_out come from the main grid, which used the same prompts and seeds; the
script verifies the overlap reproduces exactly. in_only, out_only and in_out share one
underlying string per (task, N, k, trial), so those three are paired; base_clean is paired with
base_instr (same k = 0 seed). Also reports the post-marker position on running_xor, which only
exists where the input carries markers.

Outputs: results_clean/_summary/marker_sides.csv, paper_tmlr/tables/marker_sides.tex
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
from analyze_compliance import marker_positions, expected_positions  # noqa: E402
TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}
TASKS = [("running_xor", None, None, "running\\_xor"),
         ("random_permutation_digits", "Alphabet_Size", 8, "random\\_permutation (digits)")]
ORDER = [("base_clean", "no", "no", "No marker sentence"), ("base_instr", "no", "no", "Marker sentence, no markers"),
         ("in_only", "yes", "no", "Markers in input only"), ("out_only", "no", "yes", "Markers in output only"),
         ("in_out", "yes", "yes", "Markers in input and output")]
LENGTHS, KS = [32, 64], [4, 16]


def vec(t, p):
    p = "" if pd.isna(p) else str(p)
    return np.array([i < len(p) and p[i] == c for i, c in enumerate(t)], dtype=float)


def grid(task, ac, axis, ks):
    fs = [f for f in (REPO / "results_clean" / task / "grid").glob("Qwen2.5-32B-Instruct*.csv") if "POOLED" not in f.name]
    g = pd.concat([pd.read_csv(f, dtype=TEXT, keep_default_na=False) for f in fs])
    g = g.assign(N=g.Seq_Length.astype(int), k=g.Interval.astype(int), t=g.Trial.astype(int), TAR=g.TAR.astype(float))
    g = g[g.N.isin(LENGTHS) & g.k.isin(ks) & (g.t < 20)]
    if ac:
        g = g[g[ac].astype(int) == axis]
    return g.drop_duplicates(["k", "N", "t"])


def main():
    s = pd.read_csv(REPO / "results" / "marker_sides" / "Qwen2.5-32B-Instruct.csv", dtype=TEXT, keep_default_na=False)
    s = s.assign(N=s.Seq_Length.astype(int), k=s.Interval.astype(int), t=s.Trial.astype(int), TAR=s.TAR.astype(float))
    rows = []
    for task, ac, axis, _ in TASKS:
        ref_base = grid(task, ac, axis, [0])          # base_instr
        ref_both = grid(task, ac, axis, KS)           # in_out
        for cond, has_in, has_out, _ in ORDER:
            if cond == "base_instr":
                d = ref_base
            elif cond == "in_out":
                d = ref_both
            else:
                d = s[(s.Task == task) & (s.Condition == cond)]
            post = np.nan
            if has_in == "yes":
                ref = {N: np.mean([vec(r.Ground_Truth, r.LLM_Output) for r in ref_base[ref_base.N == N].itertuples()],
                                  axis=0) for N in LENGTHS}
                pv, bv = [], []
                for r in d.itertuples():
                    v = vec(r.Ground_Truth, r.LLM_Output); idx = np.arange(len(v)); p = (idx > 0) & (idx % r.k == 0)
                    pv.append(v[p].mean()); bv.append(ref[r.N][p].mean())
                post = np.mean(pv)
            # compliance needs the raw generation; some older main-grid batches stored only the
            # cleaned answer, and those rows are excluded rather than counted as failures
            raw = d[d.Raw_Model_Output.map(lambda x: isinstance(x, str) and x != "")] \
                if "Raw_Model_Output" in d.columns else d.iloc[0:0]
            if not len(raw):
                comp = np.nan
            elif has_out == "yes":
                comp = np.mean([marker_positions(r.Raw_Model_Output) == expected_positions(len(r.Ground_Truth), r.k)
                                for r in raw.itertuples()])
            else:
                comp = np.mean([len(marker_positions(r.Raw_Model_Output)) == 0 for r in raw.itertuples()])
            rows.append(dict(task=task, cond=cond, markers_in=has_in, markers_out=has_out, n=len(d),
                             acc=d.TAR.mean(), se=d.TAR.sem(), post=post, complied=comp))
    o = pd.DataFrame(rows)
    # effects relative to the conventional baseline (marker sentence, no markers)
    for task, _, _, _ in TASKS:
        b = o[(o.task == task) & (o.cond == "base_instr")].acc.iloc[0]
        o.loc[o.task == task, "delta"] = o.loc[o.task == task, "acc"] - b
    o = o.round(4)
    o.to_csv(REPO / "results_clean" / "_summary" / "marker_sides.csv", index=False)
    print(o.to_string(index=False))
    lines = ["\\begin{tabular}{ll cc cc cc}", "\\toprule",
             "& & \\multicolumn{2}{c}{Markers} & & & \\\\", "\\cmidrule(lr){3-4}",
             "Task & Condition & Input & Output & Accuracy & Change & Followed & Post-marker \\\\", "\\midrule"]
    for task, _, _, lab in TASKS:
        for i, (cond, hi, ho, clab) in enumerate(ORDER):
            r = o[(o.task == task) & (o.cond == cond)].iloc[0]
            post = "---" if np.isnan(r.post) else f"{r.post:.2f}"
            ch = "---" if cond == "base_instr" else f"${r.delta:+.2f}$"
            comp = "n/a" if np.isnan(r.complied) else f"{r.complied:.0%}".replace("%", "\\%")
            lines.append(" & ".join(["\\task{" + lab + "}" if i == 0 else "", clab, hi, ho,
                                     f"{r.acc:.2f} $\\pm$ {1.96 * r.se:.2f}", ch, comp, post]) + " \\\\")
        if task != TASKS[-1][0]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "marker_sides.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/marker_sides.tex")


if __name__ == "__main__":
    main()
