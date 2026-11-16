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

# The anonymized mirror the TMLR submission cites; replaced by PERSON["code_url"] in the project.
ANON_REPO = "https://anonymous.4open.science/r/boundary-markers-2AF8/"


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
        # The cut must leave the short caption self-contained: track parentheses as well as
        # braces, or a ";" inside a parenthetical ends the entry on an unbalanced "(" --
        # "... through the prompt (Qwen2.5-32B." in the list of tables.
        d, p, cut = 0, 0, None
        for n, ch in enumerate(body):
            d += {"{": 1, "}": -1}.get(ch, 0)
            p += {"(": 1, ")": -1}.get(ch, 0)
            if d == 0 and p == 0 and ch in ".:;" and (n + 1 == len(body) or body[n + 1] in " \n") and n > 8:
                cut = n; break
        short = (body[:cut] + ".") if cut is not None else body
        short = short.replace("\n", " ")
        if "$" in short and short.count("$") % 2:
            short = body
        out.append(t[i:j] + "\\caption[" + short.replace("[", "{[}").replace("]", "{]}") + "]{" + body + "}")
        i = k


TEXTHEIGHT_PT = 8.8 * 72      # \textheight set by ucdavisthesis
CAPTION_PT_PER_CHAR = 0.16    # measured: a 1,629-char caption typesets to ~265pt at \small
FLOAT_SLACK_PT = 26           # \abovecaptionskip plus a margin so the float clears the footer
LONG_CAPTION_CHARS = 1200     # beyond this the caption goes down one size (see \capsize)
SMALLER_CAPTION_RATIO = 0.85  # \footnotesize against \small, by typeset height

# float_cap estimates the caption's height from its character count, which is first order: a
# caption with many short lines, wide math or \task{} runs typesets taller than the estimate.
# Where a build still reports "Float too large for page by Xpt on input line N", put that float's
# measured cap here rather than shrinking every float to cover the worst case.
CAP_OVERRIDES = {
    "tab:compliance": 0.65,   # overran by 13.2pt at the estimated 0.70
}


def float_cap(caption_chars):
    """Height budget for a float's content, as a fraction of \\textheight.

    A float is its content plus its caption, and several captions here run to 1,500-2,400
    characters. A fixed 0.70 cap on the content ignores that, so the tallest-captioned floats
    overran the bottom of the page and printed over the page number. Give the caption its
    measured height and cap the content with what is left.
    """
    rate = CAPTION_PT_PER_CHAR * (SMALLER_CAPTION_RATIO if caption_chars > LONG_CAPTION_CHARS else 1)
    budget = TEXTHEIGHT_PT - (rate * caption_chars + FLOAT_SLACK_PT)
    return max(0.35, min(0.70, budget / TEXTHEIGHT_PT))


def bound_floats(t):
    """Per float, scale the content cap to what its caption leaves on the page."""
    def one(m):
        env, blk = m.group(1), m.group(2)
        cm = re.search(r"\\caption(?:\[.*?\])?\{", blk, re.S)
        if not cm:
            return m.group(0)
        k, d = cm.end(), 1
        while d:
            d += {"{": 1, "}": -1}.get(blk[k], 0); k += 1
        nchars = k - cm.end()
        lab = re.search(r"\\label\{([^}]*)\}", blk)
        cap = CAP_OVERRIDES.get(lab.group(1) if lab else "", float_cap(nchars))
        if cap >= 0.70 and nchars <= LONG_CAPTION_CHARS:
            return m.group(0)
        if nchars > LONG_CAPTION_CHARS:
            # a 2,400-character caption is 60% of the page at \small, which leaves the table too
            # little height to stay legible; one size down buys that height back. Local to the float.
            opt = re.match(r"\s*\[[^\]]*\]", blk)
            at = opt.end() if opt else 0
            blk = blk[:at] + "\n\\renewcommand{\\capsize}{\\footnotesize}" + blk[at:]
        blk = blk.replace("max totalheight=0.70\\textheight",
                          "max totalheight=%.2f\\textheight" % cap)
        # graphics carry no height bound of their own, so a tall figure overruns the page too
        def gfx(g):
            opts = g.group(1)
            if "height" in opts:
                return g.group(0)
            return "\\includegraphics[%s,height=%.2f\\textheight,keepaspectratio]{" % (opts, cap)
        blk = re.sub(r"\\includegraphics\[([^\]]*)\]\{", gfx, blk)
        return "\\begin{%s}%s\\end{%s}" % (env, blk, env)
    return re.sub(r"\\begin\{(table|figure)\}(.*?)\\end\{\1\}", one, t, flags=re.S)


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
    # The TMLR source cites the anonymous mirror for double-blind review; the project carries the
    # author's name on its title page, so it must name the real repository instead. Fail loudly if
    # the link in the source ever changes again: an earlier version of this substitution matched a
    # \pending{...} placeholder that no longer exists, so it silently did nothing and the project
    # shipped the anonymous link.
    n_anon = (body + appendix).count(ANON_REPO)
    assert n_anon == 1, (f"expected exactly one {ANON_REPO!r} in the TMLR source, found {n_anon}; "
                         "the reproducibility link changed and this script needs updating")
    body = body.replace(ANON_REPO, PERSON["code_url"])
    appendix = appendix.replace(ANON_REPO, PERSON["code_url"])
    assert "\\pending{" not in body + appendix, "a placeholder is left in the project version"
    assert "anonymous.4open.science" not in body + appendix, "the anonymous mirror survived into the project version"
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
    body, appendix = bound_floats(body), bound_floats(appendix)
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
%% ucdavisthesis sets captions in \small. \capsize parameterizes only that size, leaving the
%% class's caption layout untouched, so a float with an unusually long caption can drop one
%% size locally and leave its table or figure enough height to stay legible.
\newcommand{\capsize}{\small}
\makeatletter
\long\def\@makecaption#1#2{%%
  \renewcommand{\baselinestretch}{1.0}%%
  \capsize
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1. #2}%%
  \ifdim \wd\@tempboxa >0.9\hsize%%
    \hskip.05\hsize\parbox{0.9\hsize}{#1. #2}%%
  \else
    \global\@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%%
  \fi
  \normalsize
  \vskip\belowcaptionskip}
\makeatother
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
