#!/usr/bin/env python3
"""Bold the cells that carry each table's claim (run after the analysis scripts and tidy_tables.py,
before compiling). One rule per table, stated in its caption in paper_tmlr/main.tex:
  wording.tex                        post-marker changes at or above zero (the blind spot removed)
  marker_sides_forced.tex            interactions whose 95% interval excludes zero
  control_cells.tex                  z values with |z| >= 3.54 (Bonferroni threshold)
Idempotent: cells already in bold are left alone."""
import re
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1] / "paper_tmlr" / "tables"
SIGNED = re.compile(r"\$([+-]\d+\.\d+)\$")


def bold_math(v):
    return "$\\mathbf{" + v + "}$"


def rows(text):
    return text.split("\n")


def real(text):
    out = []
    for line in rows(text):
        if "&" in line and "\\\\" in line and "multicolumn" not in line and "Source, task" not in line:
            line = SIGNED.sub(lambda m: bold_math(m.group(1)) if abs(float(m.group(1))) >= 0.10 else m.group(0), line)
        out.append(line)
    return "\n".join(out)


def wording(text):
    out = []
    for line in rows(text):
        if "&" in line and "\\\\" in line and ("Qwen" in line or "Gemma" in line):
            cells = line.rstrip(" \\").split("&")
            for j in range(2, 6):                                  # the four post-marker columns
                c = cells[j].strip()
                m = re.fullmatch(r"\$\+(\d+\.\d+)\$|0\.00", c)
                if m:
                    cells[j] = " " + (bold_math("+" + m.group(1)) if m.group(1) else "$\\mathbf{0.00}$") + " "
            line = "&".join(cells) + " \\\\"
        out.append(line)
    return "\n".join(out)


def sides(text):
    def rep(m):
        point, lo, hi = m.group(1), float(m.group(2)), float(m.group(3))
        return m.group(0).replace(f"${point}$", bold_math(point), 1) if lo > 0 or hi < 0 else m.group(0)
    return re.sub(r"Interaction \$([+-]\d+\.\d+)\$, 95\\% interval \$\[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\]\$", rep, text)


def control(text):
    out = []
    for line in rows(text):
        if "&" in line and "\\\\" in line and "Model &" not in line:
            cells = line.rstrip(" \\").split("&")
            for j in range(len(cells) - 3, len(cells)):            # the three z columns
                c = cells[j].strip()
                if re.fullmatch(r"-?\d+\.\d", c) and abs(float(c)) >= 3.54:
                    cells[j] = " \\textbf{" + c + "} "
            line = "&".join(cells) + " \\\\"
        out.append(line)
    return "\n".join(out)


RULES = {"wording.tex": wording,
         "marker_sides_forced.tex": sides, "control_cells.tex": control}


def main():
    for name, fn in RULES.items():
        f = TABLES / name
        if not f.exists():
            continue
        old = f.read_text()
        new = fn(old)
        if new != old:
            f.write_text(new)
        print(f"{name}: {new.count(chr(92) + 'mathbf') + new.count(chr(92) + 'textbf')} bold cells")


if __name__ == "__main__":
    main()
