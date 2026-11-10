# TMLR draft

Anonymized TMLR submission draft: *Re-Tokenization, Segmentation, and a Blind Spot: What
Content-Free Boundary Markers Do in Sequence Transformation* (`main.tex`, official TMLR style
files, `refs.bib`, `figures/`, `tables/`).

## Compile

Compiled with Tectonic (conda env `.conda/envs/tex`): `paper.pdf`. Static checks also pass (every `\ref`
resolves, every `\cite` is in `refs.bib`, every figure file exists, environments balance).

Overleaf: New Project → Upload Project → `paper_tmlr_overleaf.zip` (repo root). Compiler: pdfLaTeX,
main file `main.tex`. Rebuild the zip after edits:
`cd paper_tmlr && zip -r ../paper_tmlr_overleaf.zip . -x README.md sync_figures.sh`

## Regenerating figures and tables

All numbers come from the raw grids under `results_clean/` and `results/ablation_v2/`. From the
repository root, with `.conda/envs/qwen-focus/bin/python`:

1. `scripts/analyze_per_position.py` and `scripts/make_fig_split_heatmap.py` (summary CSVs)
2. `scripts/make_fig_three_formats.py`, `scripts/make_fig_per_position.py`
3. `scripts/make_tables_per_position.py`, `scripts/make_table_ablation_v2.py` (write `tables/`)
4. `bash paper_tmlr/sync_figures.sh`

Numbers quoted in the text are not generated automatically; if data change, recheck them against
`results_clean/_summary/*.csv` and the tables.

## Red `[...]` text

`\pending{...}` marks text still to fill. One remains: the anonymized code link.

## Before submitting

- Code link (Reproducibility paragraph): run
  `.conda/envs/qwen-focus/bin/python scripts/make_anonymous_release.py` (builds `release/anon/`, with
  an MIT license, and fails if any identifying string remains), push `release/anon/` to a new GitHub
  repository created without your name in it, mirror it at https://anonymous.4open.science, and cite
  that link.
- `refs.bib`: publication status checked 2026-11-08 against conference/journal records. Nine
  former preprints now cite their published versions (two with changed titles: MEMENTO, Hou et al.).
  Workshop-only papers (Nye et al., Nogueira et al., Wang et al.) stay as arXiv citations. Recheck
  the remaining preprints just before submitting (LEAD is under review at ICLR 2027).
- Independent review of the earlier framing: `REVIEW_2026-11-08.md`.
- For arXiv, switch to `\usepackage[preprint]{tmlr}` and add real author names.
