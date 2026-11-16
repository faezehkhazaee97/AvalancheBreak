#!/usr/bin/env python3
"""Build the UC Davis master's project version of the paper from paper_tmlr/main.tex.

The TMLR source is the single source of truth. This script turns it into paper_ms/main.tex for
the community `ucdavisthesis` class (UC Davis Graduate Studies preliminary pages, 1-inch
margins, double spacing, roman-numbered preliminaries, page numbers centered at the bottom):
sections become chapters, the abstract becomes the thesis abstract, figures, tables, macros and
the bibliography are shared through symlinks. Name, degrees, program, dates, and committee are
set in the PERSON block below; the committee lines are left blank on purpose.

Usage: python scripts/make_ms_version.py   (then compile paper_ms/main.tex with tectonic)
"""

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "paper_tmlr" / "main.tex"
OUT = REPO / "paper_ms"

PERSON = dict(
    author="Faezeh Khazaee",
    degrees="",                      # prior degrees line on the title page; empty = omitted
    major="Computer Science",        # official name of the graduate program
    month="December",
    year="2026",
    acknowledgments="",              # empty = no acknowledgments page
    code_url="https://github.com/faezehkhazaee97/AvalancheBreak",
)


def short_captions(t):
    """\\caption{long} -> \\caption[first sentence]{long}, so the lists of figures and tables stay short."""
    out, i = [], 0
    while True:
        j = t.find("\\caption{", i)
        if j < 0:
            out.append(t[i:]); return "".join(out)
        k, depth = j + len("\\caption{"), 1
        start = k
        while depth:
            depth += {"{": 1, "}": -1}.get(t[k], 0); k += 1
        body = t[start:k - 1]
        d, cut = 0, None
        for n, ch in enumerate(body):
            d += {"{": 1, "}": -1}.get(ch, 0)
            if d == 0 and ch in ".:;" and (n + 1 == len(body) or body[n + 1] in " \n") and n > 8:
                cut = n; break
        short = (body[:cut] + ".") if cut is not None else body
        short = short.replace("\n", " ")
        if "$" in short and short.count("$") % 2:
            short = body
        out.append(t[i:j] + "\\caption[" + short.replace("[", "{[}").replace("]", "{]}") + "]{" + body + "}")
        i = k


def main():
    s = SRC.read_text()
    title = re.search(r"\\title\{(.*?)\}\n\n", s, re.S).group(1).strip()
    abstract = s.split("\\begin{abstract}")[1].split("\\end{abstract}")[0].strip()
    body = s.split("\\section{Introduction}", 1)[1]
    body = "\\section{Introduction}" + body
    body, rest = body.split("\\bibliography{refs}", 1)
    appendix = rest.split("\\appendix", 1)[1].split("\\end{document}")[0]

    def chapters(t):
        t = re.sub(r"(?<!sub)\\section\{", "\\\\chapter{", t)        # top level first, so the shifted
        t = re.sub(r"(?<!sub)\\subsection\{", "\\\\section{", t)      # subsections are not promoted twice
        t = re.sub(r"\\subsubsection\{", "\\\\subsection{", t)
        # the TMLR file jumps from \section straight to \subsection; after the shift the chapter
        # command needs no change. \paragraph stays a run-in heading.
        return t

    # main sections become chapters; the appendix stays one chapter with its parts as sections,
    # so short appendices do not each start a new page
    body = chapters(body)
    appendix = "\\chapter{Supplementary Results}\n\n" + appendix
    # the TMLR source keeps a red placeholder for the anonymized code link; the project names the repository
    body = re.sub(r"\\pending\{Code, prompts, raw generations, and analysis scripts: anonymized repository link\.\}",
                  "Code, prompts, raw generations, and analysis scripts are at \\\\url{%s}." % PERSON["code_url"], body)
    assert "\\pending{" not in body + appendix, "a placeholder is left in the project version"
    # tables wider than the 6.5-inch text block are shrunk to fit; those already in a \resizebox are left alone
    # verbatim blocks and longtables cannot sit inside a box: leave those inputs alone
    fit = lambda t: re.sub(r"(?<!\{)\\input\{tables/((?!prompts_verbatim|[^}]*_lt\.tex)[^}]+)\}", r"\\adjustbox{max width=\\linewidth,max totalheight=0.70\\textheight}{\\input{tables/\1}}", t)
    body, appendix = fit(body), fit(appendix)
    # inline tabulars (Table 1, the summary and decision tables) get the same treatment
    inl = lambda t: re.sub(r"(?<!\{)(\\begin\{tabular\}.*?\\end\{tabular\})", r"\\adjustbox{max width=\\linewidth,max totalheight=0.70\\textheight}{\1}", t, flags=re.S)
    body, appendix = inl(body), inl(appendix)
    # tables the TMLR source already scales to the line width: bound their height too
    # The prompt appendix is verbatim, so it cannot go in an adjustbox; its 92-column lines
    # overrun the right margin at the thesis 12pt base size, so set it smaller instead.
    # prompts_verbatim.tex wraps each block in \begin{small}, which would override an outer
    # size, and the file is shared with the TMLR version through a symlink, so rebind \small
    # inside the group instead of editing it.
    pv = lambda t: t.replace("\\input{tables/prompts_verbatim.tex}",
                             "{\\let\\small\\scriptsize\\input{tables/prompts_verbatim.tex}}")
    # Any \resizebox width, not only \linewidth: with {!} the height scales with the width, so a
    # tall table keeps its height and runs off the page bottom unless the height is capped too.
    rb = lambda t: re.sub(r"\\resizebox\{[^}]*\}\{!\}\{",
                          r"\\adjustbox{max width=\\linewidth,max totalheight=0.70\\textheight}{", t)
    body, appendix = rb(body), rb(appendix)
    body, appendix = pv(body), pv(appendix)
    # The two position classes sit on one line, which fits the TMLR column at 10pt but runs
    # past the right margin in the thesis text block at 12pt; stack them instead.
    def stack_classes(s):
        return re.sub(r"(\\mathcal\{P\}_k.*?)\\qquad\s*(\\mathcal\{W\}_k.*?\.)",
                      "\\\\begin{aligned}\n\\1\\\\\\\\\n\\2\n\\\\end{aligned}", s, flags=re.S)
    body, appendix = stack_classes(body), stack_classes(appendix)

    # thesis wording: "the paper" -> "this work"; references to chapter-level labels say Chapter
    chap_labels = re.findall(r"\\chapter\{[^}]*\}\n\\label\{([^}]*)\}", body)
    def wording(x):
        x = re.sub(r"\b([Tt]he|[Tt]his) paper\b", lambda m: ("This" if m.group(1)[0] == "T" else "this") + " report", x)
        for lab in chap_labels:
            x = x.replace("Section~\\ref{%s}" % lab, "Chapter~\\ref{%s}" % lab)
            x = x.replace("Sections~\\ref{%s}" % lab, "Chapters~\\ref{%s}" % lab)
        return x
    body, appendix = wording(body), wording(appendix)
    body, appendix = short_captions(body), short_captions(appendix)
    assert not re.search(r"\b[Tt]h(e|is) paper\b", body + appendix)

    tex = r"""%% Generated by scripts/make_ms_version.py from paper_tmlr/main.tex. Do not edit by hand:
%% edit the TMLR source and rerun the script.
\documentclass[MS,12pt]{ucdavisthesis}
\input{math_commands.tex}
\usepackage{natbib}
\usepackage{hyperref}
\hypersetup{hidelinks}
\usepackage{url}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{placeins}
\usepackage{flafter}
\usepackage{float}
\usepackage{longtable}
\usepackage{adjustbox}
\usepackage{relsize}
\usepackage[htt]{hyphenat}
\makeatletter\setlength{\@fptop}{0pt}\makeatother
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.8}\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}\setcounter{topnumber}{3}\setcounter{bottomnumber}{2}\setcounter{totalnumber}{5}
\renewcommand{\bibfont}{\singlespacing}
\setlength{\bibsep}{4pt plus 2pt}

\newcommand{\task}[1]{\texttt{#1}}
\newcommand{\pending}[1]{\textcolor{red}{[#1]}}

\title{%(title)s}
\author{%(author)s}
\authordegrees{%(degrees)s}
\officialmajor{%(major)s}
\graduateprogram{%(major)s}
\degreemonth{%(month)s}
\degreeyear{%(year)s}
%% Major professor (chair) and committee members are left blank on purpose.
\committee{}{}{}{}{}
%% Plan II project report: the document-type line reads Project (change to Thesis for Plan I).
\makeatletter\def\@Thesisname{Project}\makeatother
%(ack_cmd)s
\abstract{%(abstract)s}

\begin{document}
\makeintropages

%(body)s
\bibliographystyle{tmlr}
\bibliography{refs}

\appendix
%(appendix)s
\end{document}
""" % dict(PERSON, title=title.replace("\\\\\n", " "), abstract=abstract, body=body.strip(), appendix=appendix.strip(),
           ack_cmd=("\\acknowledgments{%s}" % PERSON["acknowledgments"]) if PERSON["acknowledgments"] else "")
    OUT.mkdir(exist_ok=True)
    (OUT / "main.tex").write_text(tex)
    for name in ("figures", "tables", "refs.bib", "math_commands.tex", "tmlr.bst"):
        link = OUT / name
        if not link.exists():
            os.symlink(os.path.relpath(REPO / "paper_tmlr" / name, OUT), link)
    print(f"wrote {OUT / 'main.tex'} ({len(tex)} chars); title {len(title)} chars")


if __name__ == "__main__":
    main()
