# Quarantined: models never documented in overleaf/main.tex

Moved out of `results_clean/<task>/grid/` on 2026-07-27 after discovering they were being
silently swept into every glob-based aggregate (FDR correction, dose-response robustness,
completion-rate) built this session, inflating test counts / pooled numbers with data the
paper doesn't describe anywhere. See `REVIEWER_RESPONSE_NEW_RESULTS.md` section 6 for the
full incident writeup. Nothing here is deleted -- just moved so future scripts that glob
`results_clean/<task>/grid/*.csv` won't pick it up by accident.

## OLMo-2-1124-7B-Instruct -- real, complete data; write-up decision is yours

Full 9-task grid, same scope/parity as Mistral-7B-Instruct-v0.3 (matching row counts and
trial structure -- 40 trials/cell, up to N=256 for cyclic_shift). Not a rough or partial run.
Ran 2026-07-19 (slurm jobs 36215367-36216937, `slurm_logs/grid-olmo-*.log`). If you want a 6th
model in the paper, this data is ready to write up (Methods §3.3, §4.9, Appendix G) with no
further GPU time needed. If you decide not to include it, leave it quarantined here.

## Qwen2.5-72B-Instruct -- genuinely incomplete, recommend excluding regardless

Only 2 of 9 tasks (cyclic_shift, composed_permutation), 20 trials/cell, both explicitly
filenamed `_PRELIMINARY` in their own plot outputs. Ran 2026-07-16/17 (slurm jobs
36189429/36189473) -- each task took multiple GPU-hours on 2xA100 (cyclic_shift: 7h38m), which
is almost certainly why it stopped after 2 tasks rather than a data-quality issue. This was
never a complete run and there's no strong case for finishing it (72B's cost is an order of
magnitude beyond the other 5 models with no other 72B-scale coverage in the paper to compare
against). Recommend leaving this quarantined rather than writing it up or extending it.

## Contents

- `<task>/grid/OLMo-2-1124-7B-Instruct.csv` -- 9 files (full grid)
- `<task>/grid/Qwen2.5-72B-Instruct.csv` -- 2 files (cyclic_shift, composed_permutation only)
- `<task>/plots/*_72B_PRELIMINARY_*.jpg` -- 2 files, the plots that were already self-labeled
- `diagnostics_rows/<task>__<model>__rows.csv` -- 11 files, the per-trial diagnostic exports
  compute_diagnostics.py had produced for these two models before the filter was added
