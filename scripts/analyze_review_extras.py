#!/usr/bin/env python3
"""Two review follow-ups.

Echo (tasks/*_echo): after every bar the input repeats the digit just before it. Per-position
accuracy, pooled over N <= 64, k in {2, 4, 8, 16}, and moduli, at post-marker positions (i > 0,
i mod k = 0) and within-segment positions, for three runs on the same inputs: the base task without
markers (same indices), the base task with plain markers, and the echo run. Echo outputs are scored
after removing any bracketed digits the model wrote back; the share of outputs that contain one is
reported.

Natural delimiters (tasks/text_copy_previous_natural): copy-previous over the letters of WikiText
text with spaces, commas, and periods kept. Each letter after the first is classed by what separates
it from the previous letter (nothing, a space, or punctuation) and compared with the letters-only
no-marker run of text_copy_previous at the same length and letter index.

Outputs: results_clean/_summary/review_extras_echo.csv, review_extras_natural.csv,
paper_tmlr/tables/echo.tex, paper_tmlr/tables/natural_delims.tex
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RC = REPO / "results_clean"
MODELS = [("Qwen2.5 7B", "Qwen2.5-7B-Instruct"), ("Qwen2.5 32B", "Qwen2.5-32B-Instruct"), ("Gemma 4 31B", "gemma-4-31B-it")]
KS = (2, 4, 8, 16)


def read(task, tag):
    f = RC / task / "grid" / f"{tag}.csv"
    return pd.read_csv(f, dtype=str, keep_default_na=False) if f.exists() else None


def clean(s):
    return re.sub(r"\[\d\]", "", s).replace("|", "").replace(" ", "")


def per_pos(df, raw=False):
    """{(N, k, axis, trial): [correct at each index]} using the stored or the raw output."""
    out = {}
    for _, r in df.iterrows():
        truth = r.Ground_Truth.replace("|", "").replace(" ", "")
        pred = clean(r.Raw_Model_Output) if raw else r.LLM_Output.replace("|", "").replace(" ", "")
        ax = r.get("Modulus", "") if "Modulus" in df.columns else ""
        out[(int(r.Seq_Length), int(r.Interval), ax, int(r.Trial))] = [int(i < len(pred) and pred[i] == c) for i, c in enumerate(truth)]
    return out


def echo():
    rows = []
    for base in ("running_xor", "running_sum_local"):
        for label, tag in MODELS:
            e, b = read(base + "_echo", tag), read(base, tag)
            if e is None or b is None:
                continue
            E, B = per_pos(e, raw=True), per_pos(b)
            leak = e.Raw_Model_Output.str.contains(r"\[").mean()
            nomark = {}
            for (N, k, ax, t), c in B.items():
                if k == 0:
                    for i, v in enumerate(c):
                        nomark.setdefault((N, ax, i), []).append(v)
            acc = {(cond, kind): [] for cond in ("none", "plain", "echo") for kind in ("post", "within")}
            for (N, k, ax, t), ce in E.items():
                if k not in KS or (N, k, ax, t) not in B:
                    continue
                cb = B[(N, k, ax, t)]
                for i in range(1, N):
                    kind = "post" if i % k == 0 else "within"
                    acc[("echo", kind)].append(ce[i]); acc[("plain", kind)].append(cb[i])
                    acc[("none", kind)].append(np.mean(nomark.get((N, ax, i), [np.nan])))
            r = dict(task=base, model=label, n_outputs=len(e), echo_written_back=round(leak, 3))
            for (cond, kind), v in acc.items():
                r[f"{kind}_{cond}"] = round(float(np.nanmean(v)), 3) if v else None
            rows.append(r)
    if not rows:
        print("echo: no results yet"); return
    t = pd.DataFrame(rows); t.to_csv(RC / "_summary" / "review_extras_echo.csv", index=False)
    print(t.to_string(index=False))
    names = {"running_xor": "\\task{running\\_xor}", "running_sum_local": "\\task{running\\_sum\\_local}"}
    lines = ["\\begin{tabular}{ll ccc ccc c}", "\\toprule",
             "& & \\multicolumn{3}{c}{Post-marker} & \\multicolumn{3}{c}{Within segment} & \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "Task & Model & No marker & Marker & Marker + echo & No marker & Marker & Marker + echo & Echo copied \\\\", "\\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{names[r.task]} & {r.model} & {r.post_none:.2f} & {r.post_plain:.2f} & {r.post_echo:.2f} & "
                     f"{r.within_none:.2f} & {r.within_plain:.2f} & {r.within_echo:.2f} & {100 * r.echo_written_back:.0f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "echo.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/echo.tex")


def natural():
    rows = []
    for label, tag in MODELS:
        n, c = read("text_copy_previous_natural", tag), read("text_copy_previous", tag)
        if n is None or c is None:
            continue
        ctrl = {}
        for _, r in c[c.Interval == "0"].iterrows():
            truth, pred = r.Ground_Truth, r.LLM_Output
            for j, ch in enumerate(truth):
                ctrl.setdefault((int(r.Seq_Length), j), []).append(int(j < len(pred) and pred[j] == ch))
        cls = {"within": [], "space": [], "punct": []}
        for _, r in n.iterrows():
            s, N = r.Input_Given, int(r.Seq_Length)
            letters = [ch for ch in s if ch.isalpha()]
            truth = ["#"] + letters[:-1]
            pred = re.sub(r"[^a-z#]", "", r.Raw_Model_Output.lower())
            gaps = re.split(r"[a-z]", s)[1:-1]          # what lies between consecutive letters
            for j in range(1, len(letters)):
                g = gaps[j - 1]
                kind = "within" if g == "" else ("punct" if re.search(r"[,.]", g) else "space")
                ok = int(j < len(pred) and pred[j] == truth[j])
                cls[kind].append((ok, np.mean(ctrl.get((N, j), [np.nan]))))
        r = dict(model=label, n_outputs=len(n))
        for kind, v in cls.items():
            a = np.array(v, dtype=float)
            r[f"{kind}_n"] = len(a); r[f"{kind}_acc"] = round(float(a[:, 0].mean()), 3)
            r[f"{kind}_letters_only"] = round(float(np.nanmean(a[:, 1])), 3)
            r[f"{kind}_se"] = round(float(a[:, 0].std(ddof=1) / np.sqrt(len(a))), 3)
        rows.append(r)
    if not rows:
        print("natural: no results yet"); return
    t = pd.DataFrame(rows); t.to_csv(RC / "_summary" / "review_extras_natural.csv", index=False)
    print(t.to_string(index=False))
    lines = ["\\begin{tabular}{l ccc ccc}", "\\toprule",
             "& \\multicolumn{3}{c}{With spaces and punctuation} & \\multicolumn{3}{c}{Letters only, same indices} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Model & Within word & After space & After punct. & Within word & After space & After punct. \\\\", "\\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.model} & {r.within_acc:.2f} & {r.space_acc:.2f} & {r.punct_acc:.2f} & "
                     f"{r.within_letters_only:.2f} & {r.space_letters_only:.2f} & {r.punct_letters_only:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "paper_tmlr" / "tables" / "natural_delims.tex").write_text("\n".join(lines) + "\n")
    print("wrote tables/natural_delims.tex")


if __name__ == "__main__":
    echo()
    natural()
