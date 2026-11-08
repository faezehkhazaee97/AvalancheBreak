# TMLR draft

Anonymized TMLR submission draft (`main.tex`, official TMLR style files, `refs.bib`, `figures/`).

## Compile

There is no LaTeX on Farm, so this has not been compiled yet; static checks pass (every `\ref`
resolves, every `\cite` is in `refs.bib`, every figure file exists, environments balance).

Overleaf: New Project → Upload Project → `paper_tmlr_overleaf.zip` (repo root). Compiler: pdfLaTeX, main file `main.tex`.
Rebuild the zip after edits: `cd paper_tmlr && zip -r ../paper_tmlr_overleaf.zip . -x README.md sync_figures.sh`

## Red `[...]` text

`\pending{...}` marks text that depends on runs still in progress (Qwen3.8-27B, Gemma-4-31B-it,
Qwen2.5-32B second batches). Search for `\pending` and resolve each before submitting.

## When new results land

1. `.conda/envs/qwen-focus/bin/python scripts/make_fig_split_heatmap.py` (Figure 1 picks up new
   models and batches automatically; numbers go to `results_clean/_summary/split_heatmap_values.csv`),
   then `scripts/make_fig3_causal_only.py` (the reset figure reads that CSV).
2. `bash paper_tmlr/sync_figures.sh`
3. Update the numbers quoted in Sections 4–5 and Tables `tab:reset` / `tab:trials` from that CSV.

## Before submitting

- Code link (Reproducibility paragraph): after the k=1 rerun is merged and committed, run
  `.conda/envs/qwen-focus/bin/python scripts/make_anonymous_release.py` (builds `release/anon/` and fails if
  any identifying string remains), push `release/anon/` to a new GitHub repository created without your
  name in it, mirror it at https://anonymous.4open.science, and cite that link.
- `refs.bib`: publication status checked 2026-11-08 against conference/journal records. Nine
  former preprints now cite their published versions (two with changed titles: MEMENTO, Hou et al.).
  Workshop-only papers (Nye et al., Nogueira et al., Wang et al.) stay as arXiv citations. Recheck
  the remaining preprints just before submitting (LEAD is under review at ICLR 2027).
- For arXiv, switch to `\usepackage[preprint]{tmlr}` and add real author names.
