#!/usr/bin/env python3
"""Typographic pass over the generated tables in paper_tmlr/tables/ (run after the analysis scripts,
before compiling): signed zeros (+0.00, -0.00, $+0.00$, $-0.00$) become 0.00, and a bare ASCII
hyphen in front of a number outside math mode becomes a math minus sign. Only text outside $...$
is rewritten, so math already typeset is left alone."""
import re
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1] / "paper_tmlr" / "tables"
ZERO_MATH = re.compile(r"\$[+-]0\.0+\$")
ZERO_BARE = re.compile(r"(?<![\w.$])[+-](0\.0+)(?![\d])")
NEG_BARE = re.compile(r"(?<![\w.$\-])-(\d+\.\d+)(?![\w$])")


def tidy_line(line):
    line = ZERO_MATH.sub(lambda m: "0." + "0" * (len(m.group(0)) - 5), line)
    parts = line.split("$")
    for i in range(0, len(parts), 2):          # even segments are outside math
        parts[i] = ZERO_BARE.sub(r"\1", parts[i])
        parts[i] = NEG_BARE.sub(r"$-\1$", parts[i])
    return "$".join(parts)


def main():
    changed = 0
    for f in sorted(TABLES.glob("*.tex")):
        old = f.read_text()
        new = "\n".join(tidy_line(l) if not l.lstrip().startswith("%") else l for l in old.split("\n"))
        if new != old:
            f.write_text(new); changed += 1
    print(f"tidied {changed} table files")


if __name__ == "__main__":
    assert tidy_line("a & -0.00 & $+0.00$ & -0.53 & $-0.12$ & (0.28 at 13)") == "a & 0.00 & 0.00 & $-0.53$ & $-0.12$ & (0.28 at 13)", tidy_line("a & -0.00 & $+0.00$ & -0.53 & $-0.12$ & (0.28 at 13)")
    main()
