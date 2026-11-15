#!/usr/bin/env python3
"""Bold the cells that carry each table's claim (run after the analysis scripts and tidy_tables.py,
before compiling). One rule per table, stated in its caption in paper_tmlr/main.tex:
  wording.tex                        post-marker changes at or above zero (the blind spot removed)
  marker_sides_forced.tex            interactions whose 95% interval excludes zero, and
  marker_sides_forced_full.tex       (post - within) - base values whose 95% interval excludes zero
  marker_sides_forced_7b.tex         (post - within) - base values whose 95% interval excludes zero
  control_cells.tex                  z values with |z| >= 3.54 (Bonferroni threshold)
Further rules (real_data, ablation_32b, perpos, aligned, blind_spot_by_k, blind_spot_window, patching,
attention_crossing, lora) are listed in RULES and stated in each caption. Idempotent: cells already in bold
are left alone."""
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


DID = re.compile(r"(\$[+-]\d+\.\d+\$|(?<![\w.])0\.\d+(?![\w.])) \$\[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\]\$")


def sides_did(text):
    """Post-marker minus within-segment, relative to the no-marker cell: bold the point estimate
    where the 95% interval excludes zero. The interaction line is left to sides(): its point and
    interval are separated by ', 95\\% interval ', so this pattern cannot match it."""
    def rep(m):
        point, lo, hi = m.group(1), float(m.group(2)), float(m.group(3))
        return m.group(0).replace(point, bold_math(point.strip("$")), 1) if lo > 0 or hi < 0 else m.group(0)
    return DID.sub(rep, text)


def control(text):
    out = []
    for line in rows(text):
        if "&" in line and "\\\\" in line and "Model &" not in line:
            cells = line.rstrip(" \\").split("&")
            for j in (7, 8, 9):                                    # the three z columns
                c = cells[j].strip()
                # a negative z arrives as $-3.8$ (tidy_tables.py has already made the minus a
                # math minus), so match it with or without the dollars and bold it in kind
                m = re.fullmatch(r"\$?(-?\d+\.\d)\$?", c)
                if m and abs(float(m.group(1))) >= 3.54:
                    cells[j] = (" $\\mathbf{" + m.group(1) + "}$ " if c.startswith("$")
                                else " \\textbf{" + m.group(1) + "} ")
            line = "&".join(cells) + " \\\\"
        out.append(line)
    return "\n".join(out)


NUM = re.compile(r"\$?([+-]?\d+\.\d+|\d+)\$?(\\%)?")


def bold_cells(text, cols, pred, row_filter=None):
    """Bold the first number in each listed column of every data row (rows after the first \midrule)
    when pred(value) holds; numbers already in bold are skipped."""
    out, data = [], False
    for line in text.split("\n"):
        if line.strip() == "\\midrule":
            data = True
        if data and "&" in line and line.rstrip().endswith("\\\\") and "multicolumn" not in line and (row_filter is None or row_filter(line)):
            cells = line.rstrip()[:-2].split("&")
            for j in cols:
                if j >= len(cells) or "mathbf" in cells[j] or "textbf" in cells[j]:
                    continue
                m = NUM.search(cells[j])
                if not m:
                    continue
                v = float(m.group(1))
                if pred(v):
                    tok = m.group(0)
                    if tok.startswith("$"):
                        new = "$\\mathbf{" + m.group(1) + "}" + ("\\%" if m.group(2) else "") + "$"
                    else:
                        new = "\\textbf{" + m.group(1) + ("\\%" if m.group(2) else "") + "}"
                    cells[j] = cells[j][:m.start()] + new + cells[j][m.end():]
            line = "&".join(cells) + "\\\\"
        out.append(line)
    return "\n".join(out)


RULES = {"wording.tex": wording,
         "marker_sides_forced.tex": lambda t: sides_did(sides(t)),
         "marker_sides_forced_full.tex": lambda t: sides_did(sides(t)),
         "marker_sides_forced_7b.tex": sides_did,
         "control_cells.tex": control,
         # merged real-sequence tables: column 7 (0-based 6) is the index-scored post-marker change
         "real_data.tex": lambda t: bold_cells(t, [6], lambda v: abs(v) >= 0.25),
         "real_data_full.tex": lambda t: bold_cells(t, [6], lambda v: abs(v) >= 0.25),
         "ablation_32b.tex": lambda t: bold_cells(t, [7], lambda v: v >= 10),
         "perpos.tex": lambda t: bold_cells(t, [6], lambda v: abs(v) >= 0.20),
         "aligned.tex": lambda t: bold_cells(t, [3], lambda v: v <= -0.10),
         # k = 2, 3, 4, 5, 6, 8, 16: the seven post-marker columns are 0-based 2..8 (within-segment 9..15)
         "blind_spot_by_k.tex": lambda t: bold_cells(t, [2, 3, 4, 5, 6, 7, 8], lambda v: v <= -0.25),
         # same seven spacings then the pooled column, 0-based 9
         "blind_spot_window.tex": lambda t: bold_cells(t, [9], lambda v: v <= -0.10),
         "patching.tex": lambda t: bold_cells(t, [5, 6, 7, 8], lambda v: v >= 0.50),
         "attention_crossing.tex": lambda t: bold_cells(t, [3], lambda v: v <= 0.20),
         "lora.tex": lambda t: bold_cells(t, [8, 11], lambda v: v >= 0.0)}


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
