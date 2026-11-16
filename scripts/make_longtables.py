#!/usr/bin/env python3
"""Derived table variants: the dependency block of the per-position table on its own, and a
shortened compliance table. Run after the analysis scripts, before compiling.

A longtable wrapper used to live here for perpos_tasks, but nothing input the result and
__main__ never called it, so the file it produced went stale; both are gone.."""
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1] / "paper_tmlr" / "tables"


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
