#!/usr/bin/env python3
"""What does the baseline's marker sentence cost? (scripts/run_clean_baseline.py)

For every (model, task, length, alphabet) cell the clean no-marker run (prompt never mentions
markers) is compared with the paper's no-marker run (prompt keeps the marker sentence) on the
same inputs: both use the k = 0 seed, so this is a paired comparison of identical prompts
differing only in the presence of that sentence. Reports the cost per model and task family,
with a 95% interval over cells, and the headline Finding-2 quantities (gain and share of
remaining errors removed at N = 64, spaced letters and digits) recomputed against the clean
baseline.

Outputs: results_clean/_summary/clean_baseline.csv, paper_tmlr/tables/clean_baseline.tex
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load  # noqa: E402

TEXT = {c: str for c in ("Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output")}
MODELS = [("Qwen2.5 14B", "Qwen2.5-14B-Instruct"), ("Qwen2.5 32B", "Qwen2.5-32B-Instruct"),
          ("Qwen3.8 27B", "Qwen3.8-27B"), ("Gemma 4 31B", "gemma-4-31B-it")]
FAM = {"spaced letters": ["cyclic_shift_spaced", "random_permutation_spaced", "conditional_shift_spaced"],
       "digits": ["cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits"],
       "dependency": ["copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]}
LENGTHS = [16, 32, 64]


def main():
    cells, head = [], []
    for label, tag in MODELS:
        f = REPO / "results" / "clean_baseline" / f"{tag}.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f, dtype=TEXT, keep_default_na=False)
        c = c.assign(N=c.Seq_Length.astype(int), axis=c.Axis.astype(int), t=c.Trial.astype(int),
                     TAR=c.TAR.astype(float))
        for fam, tasks in FAM.items():
            for task in tasks:
                g = load(task, tag)
                if g is None:
                    continue
                ac = next((x for x in ("Alphabet_Size", "Modulus") if x in g.columns), None)
                g = g.assign(N=g.Seq_Length.astype(int), k=g.Interval.astype(int), t=g.Trial.astype(int),
                             TAR=g.TAR.astype(float), axis=g[ac].astype(int) if ac else 2)
                base = g[(g.k == 0) & g.N.isin(LENGTHS)].drop_duplicates(["N", "axis", "t"])
                cc = c[c.Task == task]
                for (N, ax), grp in cc.groupby(["N", "axis"]):
                    b = base[(base.N == N) & (base.axis == ax)].set_index("t").TAR
                    paired = grp.set_index("t").TAR.to_frame("clean").join(b.rename("instr"), how="inner")
                    if len(paired) < 5:
                        continue
                    cells.append(dict(model=label, family=fam, task=task, N=N, axis=ax, n=len(paired),
                                      clean=paired.clean.mean(), instr=paired.instr.mean(),
                                      cost=(paired.clean - paired.instr).mean()))
                # Finding-2 recomputation at N = 64 against the clean baseline
                if fam != "dependency":
                    bnd = g[(g.k.isin([1, 2, 4, 8, 16])) & (g.N == 64)]
                    cl = cc[cc.N == 64]
                    bi = base[base.N == 64]
                    if len(bnd) and len(cl) and len(bi):
                        head.append(dict(model=label, family=fam, task=task, bounds=bnd.TAR.mean(),
                                         base_instr=bi.TAR.mean(), base_clean=cl.TAR.mean()))
    o = pd.DataFrame(cells).round(4)
    o.to_csv(REPO / "results_clean" / "_summary" / "clean_baseline.csv", index=False)
    print(f"{len(o)} paired cells across {o.model.nunique()} models")
    s = o.groupby(["model", "family"]).cost.agg(["mean", "sem", "count"]).round(3)
    s["ci"] = (1.96 * s["sem"]).round(3)
    print(s.to_string())
    print("\noverall cost (clean - with sentence): %+.3f, 95%% [%+.3f, %+.3f] over %d cells"
          % (o.cost.mean(), o.cost.mean() - 1.96 * o.cost.sem(), o.cost.mean() + 1.96 * o.cost.sem(), len(o)))
    print("cells where the sentence HELPS (cost < 0): %d of %d" % ((o.cost < 0).sum(), len(o)))
    h = pd.DataFrame(head)
    if len(h):
        h["gain_instr"] = h.bounds - h.base_instr
        h["gain_clean"] = h.bounds - h.base_clean
        h["share_instr"] = h.gain_instr / (1 - h.base_instr)
        h["share_clean"] = h.gain_clean / (1 - h.base_clean)
        print("\nFinding 2 at N = 64, per model (mean over the six control cells):")
        print(h.groupby("model")[["gain_instr", "gain_clean", "share_instr", "share_clean"]].mean().round(3).to_string())
        h.round(4).to_csv(REPO / "results_clean" / "_summary" / "clean_baseline_finding2.csv", index=False)
    # table: cost by model and family
    lines = ["\\begin{tabular}{l ccc c}", "\\toprule",
             "Model & Spaced letters & Digits & Dependency & All \\\\", "\\midrule"]
    for label, _ in MODELS:
        x = o[o.model == label]
        if x.empty:
            continue
        cells_ = []
        for fam in ("spaced letters", "digits", "dependency"):
            y = x[x.family == fam].cost
            cells_.append("---" if y.empty else f"${y.mean():+.3f} \\pm {1.96 * y.sem():.3f}$")
        cells_.append(f"${x.cost.mean():+.3f} \\pm {1.96 * x.cost.sem():.3f}$")
        lines.append(f"{label} & " + " & ".join(cells_) + " \\\\")
    lines += ["\\midrule", f"All & & & & ${o.cost.mean():+.4f} \\pm {1.96 * o.cost.sem():.4f}$ \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "clean_baseline.tex").write_text("\n".join(lines) + "\n")
    print("Wrote paper_tmlr/tables/clean_baseline.tex")


if __name__ == "__main__":
    main()
