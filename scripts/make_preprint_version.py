#!/usr/bin/env python3
"""Build the named preprint (arXiv) version of the paper from paper_tmlr/main.tex.

The TMLR source is anonymized for double-blind review: the author block reads "Anonymous
Authors" and the reproducibility link points at the anonymous mirror. Neither is right for a
preprint, and editing them in place would break the submission version, so this script derives
paper_arxiv/main.tex instead, exactly as scripts/make_ms_version.py derives the project version.

Three changes, nothing else:
  1. \\usepackage{tmlr} -> \\usepackage[preprint]{tmlr}, the style's own switch for a named
     version (it sets \\@accepted, which prints the author block instead of suppressing it).
  2. the anonymous author block -> the real name, email and affiliation in PERSON below.
  3. the anonymous.4open.science mirror -> the real repository.

Figures, tables, the bibliography and the macros are shared with paper_tmlr through symlinks, so
the preprint never drifts from the submission.

Usage: python scripts/make_preprint_version.py   (then compile paper_arxiv/main.tex with tectonic)
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "paper_tmlr" / "main.tex"
OUT = REPO / "paper_arxiv"

PERSON = dict(
    name="Faezeh Khazaee",
    email="fkhazaee@ucdavis.edu",
    affiliation="Department of Computer Science \\\\ University of California, Davis",
    repo="https://github.com/faezehkhazaee97/AvalancheBreak",
)

ANON_AUTHOR = ("\\author{\\name Anonymous Authors \\email anonymous@example.com \\\\\n"
               "      \\addr Anonymous Institution}")
ANON_REPO = "https://anonymous.4open.science/r/boundary-markers-2AF8/"


def main():
    t = SRC.read_text()

    subs = [
        ("\\usepackage{tmlr}", "\\usepackage[preprint]{tmlr}"),
        (ANON_AUTHOR,
         "\\author{\\name %(name)s \\email %(email)s \\\\\n      \\addr %(affiliation)s}" % PERSON),
        (ANON_REPO, PERSON["repo"]),
    ]
    for old, new in subs:
        n = t.count(old)
        if n != 1:
            raise SystemExit(f"expected exactly one occurrence of {old[:48]!r}, found {n}; "
                             "the TMLR source changed and this script needs updating")
        t = t.replace(old, new)

    for bad in ("Anonymous", "anonymous@example.com", "anonymous.4open.science"):
        assert bad not in t, f"{bad!r} still present in the preprint version"

    OUT.mkdir(exist_ok=True)
    (OUT / "main.tex").write_text(t)
    for name in ("figures", "tables", "refs.bib", "math_commands.tex", "tmlr.bst", "tmlr.sty",
                 "fancyhdr.sty"):
        link = OUT / name
        src = REPO / "paper_tmlr" / name
        if not link.exists() and src.exists():
            os.symlink(os.path.relpath(src, OUT), link)
    print(f"wrote {OUT / 'main.tex'} ({len(t)} chars); author: {PERSON['name']}")


if __name__ == "__main__":
    main()
