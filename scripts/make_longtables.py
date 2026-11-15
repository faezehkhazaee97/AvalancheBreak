#!/usr/bin/env python3
"""Wrap generated tables that are too long for one page as longtables (header repeated on every page).
Reads paper_tmlr/tables/<name>.tex (a plain tabular) and writes paper_tmlr/tables/<name>_lt.tex.
Run after the analysis scripts, before compiling."""
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1] / "paper_tmlr" / "tables"
LONG = {
    "perpos_tasks": ("tab:perpostasks",
                     "Per-position accuracy by task and model ($N \\le 64$, $k \\in \\{2,4,8,16\\}$): the no-marker run's "
                     "accuracy at the same position indices, the marker run's accuracy, and the change, within segments and "
                     "right after a marker. These are the per-task values behind Table~\\ref{tab:perpos} and "
                     "Figure~\\ref{fig:position}."),
}


def convert(name, label, caption):
    lines = (TABLES / f"{name}.tex").read_text().strip().split("\n")
    spec = lines[0][len("\\begin{tabular}"):]
    mid = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule")
    head = lines[1:mid + 1]
    body = [l for l in lines[mid + 1:] if l.strip() not in ("\\bottomrule", "\\end{tabular}")]
    out = (["\\begin{longtable}" + spec, f"\\caption{{{caption}}}\\label{{{label}}}\\\\"] + head + ["\\endfirsthead"]
           + [f"\\multicolumn{{{spec.count('l') + spec.count('c') + spec.count('r')}}}{{l}}{{\\emph{{Table~\\ref{{{label}}}, continued}}}}\\\\"]
           + head + ["\\endhead", "\\bottomrule", "\\endlastfoot"] + body + ["\\end{longtable}"])
    (TABLES / f"{name}_lt.tex").write_text("\n".join(out) + "\n")
    print(f"wrote tables/{name}_lt.tex ({len(body)} body lines)")


def dependency_block():
    """The dependency-task block of perpos_tasks.tex on its own, as an ordinary tabular (one page)."""
    lines = (TABLES / "perpos_tasks.tex").read_text().strip().split("\n")
    mid = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule")
    start = next(i for i, l in enumerate(lines) if "Dependency tasks" in l)
    body = [l for l in lines[start + 1:] if l.strip() not in ("\\bottomrule", "\\end{tabular}")]
    (TABLES / "perpos_tasks_dep.tex").write_text("\n".join(lines[:mid + 1] + body + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    print(f"wrote tables/perpos_tasks_dep.tex ({len(body)} rows)")


def compliance_short():
    """compliance.tex with two rows per model (dependency tasks, all groups) and the full pooled block."""
    lines = (TABLES / "compliance.tex").read_text().strip().split("\n")
    mid = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule")
    out, block = lines[:mid + 1], []

    def flush(b):
        if not b:
            return []
        name = b[0].split("&")[0].strip()
        if name.startswith("All models"):
            return b
        keep = [l for l in b if "Dependency" in l or "all groups" in l]
        keep = [" & ".join([name if i == 0 else ""] + l.split("&")[1:]).replace("  ", " ") if True else l for i, l in enumerate(keep)]
        return keep

    for l in lines[mid + 1:]:
        if l.strip() in ("\\midrule", "\\bottomrule", "\\end{tabular}"):
            out += flush(block); block = []
            out.append(l)
        else:
            block.append(l)
    (TABLES / "compliance_short.tex").write_text("\n".join(out) + "\n")
    print(f"wrote tables/compliance_short.tex ({len(out)} lines)")


if __name__ == "__main__":
    dependency_block()
    compliance_short()
