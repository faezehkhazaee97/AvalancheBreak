# Q7 (cheap half): does boundary density compound early termination?

Computed from diagnostic row-level data (`results_clean/_diagnostics/*__rows.csv`,
Completion Rate = fraction of trials whose output is at least as long as expected), broken
out by `Interval` (k) instead of pooled Base-vs-Bounds. 43 task×model files (9 tasks x the 5
models documented in `overleaf/main.tex`; Phi-3.5-mini-instruct is missing 2 tasks, matching
its documented reduced scope), k=3 excluded throughout (consistent with the rest of the study).
Code: `tasks/common/compute_completion_rate.py`.

**CORRECTION (2026-07-27): the original table below was computed from 54 files, 11 of which
were OLMo-2-1124-7B-Instruct and Qwen2.5-72B-Instruct — two models never described in
`main.tex` (see the FDR-correction section's provenance note). Same root cause as the FDR
family-size bug: `select_canonical_files()`-style directory globbing swept in leftover runs.
Fixed at the source (`compute_diagnostics.py`'s `DOCUMENTED_MODELS` filter) and the stale
rows files moved to `results_clean/_diagnostics/_excluded_undocumented_models/`. Numbers below
are the corrected, 5-model version — use these, not any cached copy of the old table.**

## Pooled result (across all tasks/models) — not simply monotonic

| k | density (1/k) | pooled CR |
|---|---|---|
| Base | 0 | 0.846 |
| 16 | 0.062 | 0.748 |
| 8 | 0.125 | 0.860 |
| 4 | 0.250 | **0.869** (peak) |
| 2 | 0.500 | 0.820 |
| 1 (densest) | 1.000 | 0.732 |

Same qualitative shape as before correction: completion rate is worst at both extremes
(sparsest k=16 and densest k=1), best at moderate density (k=4), and every density still beats
neither-extreme's low point. Not the clean "more boundaries -> more early stopping" story the
question implies at the pooled level. (Absolute CR values shifted upward by ~0.08-0.1 across
the board vs. the uncorrected table — OLMo had a lower baseline completion rate than the 5
documented models, which was dragging the pooled numbers down.)

## Per-task correlation between density and CR — now unanimous, not 7-of-9

| Task | corr(density, CR) |
|---|---|
| cyclic_shift | -0.412 |
| copy_previous | -0.369 |
| running_xor | -0.326 |
| conditional_shift | -0.166 |
| random_permutation | -0.138 |
| character_substitution | -0.132 |
| composed_permutation | -0.098 |
| running_max | -0.069 (~flat) |
| running_sum_local | -0.044 (~flat) |

**This changes the qualitative claim, not just the numbers.** Under the corrected 5-model set,
all 9 of 9 tasks show a negative correlation between boundary density and completion rate —
the "7 of 9, with `running_sum_local` reversing" framing from the uncorrected table is gone;
`running_sum_local`'s reversal was an OLMo-driven artifact, not a real property of the task.
The strongest relationships (`cyclic_shift`, `copy_previous`, `running_xor`) are still
moderate, not strong (all now <0.5 in magnitude — the old "-0.808 strong" for `running_xor` was
also partly an OLMo effect and drops to -0.326 here), and `running_max`/`running_sum_local`
are both effectively flat. Honest read: the effect is real, directionally universal across
tasks in this corrected data, but modest in magnitude and clearly not the dominant driver of
completion failure — worth stating in those terms, not as either "confirmed" or "task-dependent."

Data: `results_clean/_diagnostics/completion_rate_by_density.csv`
