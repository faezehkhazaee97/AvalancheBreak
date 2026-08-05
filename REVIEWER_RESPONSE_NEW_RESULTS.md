# New experiments run in response to reviewer questions

This file summarizes code + experiments run to answer specific reviewer questions on the
draft. Everything below is backed by real data sitting in `results/` or `results_clean/`.
Give this file to Claude-writing-the-paper and ask it to (a) add the new findings to the
relevant sections/appendices, (b) fix the two factual discrepancies flagged near the bottom,
and (c) apply the reset-variant second-batch numbers (section 14 below, now complete) to
verify §4.5's single-batch caveat and drop the corresponding Limitations bullet.

**Scorecard against the review's "Questions for Authors," for quick reference:**

| # | Question | Status |
|---|---|---|
| 1 | Dose-response model selection (AIC/BIC/CV/vertex CIs) | Done -- section 7 below, with an important caveat |
| 2 | Multiple comparisons / FDR | Done -- section 6 below |
| 3 | Ablation robustness (log-probs, mean-replacement, head stability) | Done -- sections 4 and 9 below |
| 4 | Token identity / placement jitter | Done -- identity was pre-existing (Appendix B), jitter is section 10 below |
| 5 | External tasks (non-trivial CLRS competence + semi-natural local task) | Done -- section 2 below (vowel tagging) + section 11 below (ProcBench/string-copy) + separately, `clrs_minimum` at 32B (not in this file, already in the report) |
| 6 | Practical k-tuning recipe + SK params vs. density | Done -- sections 1 and 7 below, same CV caveat as #1 |
| 7 | Well-formedness: compounding early termination (done, section 8) + can prompt engineering recover WFR (**not run, explicitly deferred as future work**) | Half done |

Still open: the **Gemini 10-trial rerun** (paused, not pursued further per explicit
instruction -- leave as a stated limitation, not a result to report). The **reset-variant
second batch** is now complete (section 14 below) -- no longer open.

---

## 1. Binomial-likelihood MLE SK fit (answers: "did you try a binomial MLE instead of
least-squares, and do the alpha trends persist?")

**Why**: SAR(N) is a proportion of trials that were fully correct — a binomial observation,
not a continuous measurement with Gaussian noise. The existing fit minimizes sum-of-squares
on the *aggregated* SAR(N) points; this instead maximizes the binomial log-likelihood over
the *raw per-trial* 0/1 outcomes directly (statistically the correct likelihood for this
data), with 500-iteration bootstrap CIs and explicit flagging of bound-pinned (uninformative)
fits.

**Code**: `scripts/run_sk_fit_binomial_mle.py` (new, CPU-only, no GPU needed)
**Data**: `results/Qwen2.5-7B-Instruct/thermodynamics_sk_fit_binomial_mle.csv`

**Result — the alpha trends persist under the statistically correct likelihood**:

| Alphabet | Method | Interval | Alpha (original LSQ) | Alpha (binomial MLE) | Pinned at bound? |
|---|---|---|---|---|---|
| 2 | Base | 0 | 1.212 | 1.212 | No |
| 2 | Cellular_Bounds | k=1 | 1.037 | 1.051 | No |
| 2 | Cellular_Bounds | k=2 | — | 1.011 | No |
| 4 | Base | 0 | — | 1.717 | No |
| 4 | Cellular_Bounds | k=1 | — | 1.112 | No |
| 8 | Base | 0 | — | 1.480 | No |
| 8 | Cellular_Bounds | k=1 | — | 1.061 | No |

Base alpha is consistently higher (1.21–1.72) than dense-boundary alpha (1.01–1.11) under
both fitting methods. Some k=2/k=3 cells at larger alphabets pin at the lower bound
(alpha≈1.000001) under the MLE and are explicitly flagged `Pinned_At_Bound=True` — these
should be reported as uninformative, not as evidence of anything.

**Suggested placement**: Appendix D, as a robustness check alongside the existing bootstrap/AIC material.

---

## 2. Real, position-independent task: vowel/consonant tagging (answers: "do boundaries help
on any real task, even small-scale?")

**Why**: every task in the main study is synthetic. This tests a genuinely real,
position-independent task: tag each character in a random string as VOWEL or CONSONANT
(each character's correct tag depends only on itself — no cross-position dependency at all).

**Code**: `scripts/run_real_task_vowel_tagging.py` (pre-existing, was written but never
run before this pass) + `scripts/run_real_task_vowel_tagging_32b.py` (new, same script
targeting Qwen2.5-32B-Instruct)
**Data**: `results/Qwen2.5-7B-Instruct/vowel_tagging_boundary_sweep.csv`,
`results/Qwen2.5-32B-Instruct/vowel_tagging_boundary_sweep.csv`
**Grid**: N ∈ {8,16,32,64}, k ∈ {0,2,4,8}, 10 trials/cell

**Result — mixed at 7B, clean positive at 32B**:

| N | 7B Base | 7B Best (k) | 7B Gap | 32B Base | 32B Best (k) | 32B Gap |
|---|---|---|---|---|---|---|
| 8 | 0.538 | 0.538 (k=0) | +0.000 | 0.675 | 0.750 (k=4) | +0.075 |
| 16 | 0.438 | 0.463 (k=4) | +0.025 | 0.594 | 0.694 (k=2) | +0.100 |
| 32 | 0.472 | 0.472 (k=0) | +0.000 | 0.578 | 0.759 (k=4) | +0.181 |
| 64 | 0.330 | 0.336 (k=2) | +0.006 | 0.577 | 0.625 (k=4) | +0.048 |

At 7B the effect is essentially flat/near-zero. Checked raw outputs directly — this is a
genuine capability-floor failure, not noise or a scoring bug: the model frequently falls
into a degenerate `CVCVCVCV`-style alternating pattern regardless of the actual input
letters (e.g. input `YFOFPGPU`, expected `CCVCCCCV`, predicted `CVCVCVCV`), the same shape
of failure as `composed_permutation`'s capability wall in the main study. At 32B, where Base
competence is real (0.58–0.68 vs 7B's 0.33–0.54), boundaries help consistently and
substantially at every length (+0.048 to +0.181).

**Suggested placement**: Either a new subsection under 3.10 (External validity) alongside
CLRS-Text/Gemini, or answering the reviewer's real-task question directly in the response
letter with a forward reference to a new appendix table. Worth being explicit that the 7B
null is itself informative (competence-gating generalizes to a real task, not just the
synthetic ones) rather than a contradiction.

---

## 3. Addition state-carrying checkpoint (answers: "does a state-carrying boundary variant
narrow the gap on addition?")

**Why**: modsum already has a state-carrying checkpoint result (Section 3.7: replacing "|"
with the true running-sum value at each boundary helps at the harder modulus). A literal
port of that idea to addition is **not well-posed** in addition's normal (most-significant-
digit-first) presentation, because addition's dependency — the carry bit — propagates
right-to-left, opposite to both reading order and generation order. A checkpoint in MSD
order would have to state something not yet determined by anything the model has seen.

The correctly-posed version combines two things already separately validated in this
project: (1) present digits least-significant-digit-first, so the carry's true dependency
direction matches reading order (already tested alone as the LSD-reversal condition, and
did not rescue Cellular_Bounds); (2) on top of LSD order, insert a genuine checkpoint
stating the TRUE carry bit (0 or 1) generated by every digit pair before it — which, in LSD
order, *is* fully determined by what's already been shown, exactly like modsum's checkpoint.
Digits are presented as interleaved (a_i, b_i) pairs.

**Code**: `scripts/run_addition_state_carrying.py` (new). Engine logic independently
verified against a from-scratch most-significant-digit-first addition before running on GPU.
**Data**: `results/Qwen2.5-7B-Instruct/addition_state_carrying_grid.csv`,
`results/Qwen2.5-7B-Instruct/addition_state_carrying_comparison.csv`
**Grid**: N ∈ {8,16,32,64}, interval=4 (matches modsum's state-carrying interval), 20 trials/cell

**Result — modest, real improvement, nowhere close to closing the gap**:

| N | Base (LSD) | Cellular, content-free | Cellular, state-carrying |
|---|---|---|---|
| 8 | 0.356 | 0.081 | 0.119 |
| 16 | 0.359 | 0.088 | 0.091 |
| 32 | 0.291 | 0.081 | 0.084 |
| 64 | 0.168 | 0.080 | 0.091 |

The state-carrying checkpoint beats the content-free boundary at every length, but the
improvement is small (roughly +0.01 to +0.04 TAR) and Base still dominates by a wide margin
throughout. Honest reading: state-carrying checkpoints help *some*, consistent with the
modsum finding, but addition's carry-chain dependency is harder to rescue this way than
modsum's running sum — plausibly because addition needs the checkpoint to be exactly right
at every single position (any wrong carry corrupts everything after it) in a way modsum's
looser dependency doesn't.

**Suggested placement**: Appendix C, as a direct answer to the "does a state-carrying
variant narrow the gap" reviewer question, framed as informative-negative (helps a little,
doesn't rescue) rather than a clean win.

---

## 4. Head stability across seeds/lengths/densities (answers: "how stable are the identified
sink heads — is there a persistent subset, or does the set drift?")

**Why**: the existing causal-ablation head identification (Section 4.2/4.3) was done once,
under one condition (N=16, k=2, alphabet=2, trials 0-4). This reruns the identical
identification procedure under 7 different conditions and measures overlap against that
original reference set.

**Code**: `scripts/run_head_stability_check.py` (new). Reuses the exact
`find_boundary_token_indices` crop and top-10-by-mean-boundary-attention-mass selection
from `scripts/run_causal_ablation.py`, unchanged, so results are directly comparable.
**Data**: `results/Qwen2.5-7B-Instruct/head_stability_summary.csv`

**Result — a real 3-head persistent core, with drift beyond that**:

| Condition | Overlap with reference (of 10) | Jaccard |
|---|---|---|
| Reference (N=16, k=2, trials 0-4) | 10 | 1.00 |
| Different seeds (trials 5-9) | 8 | 0.67 |
| Different seeds (trials 10-14) | 7 | 0.54 |
| Different length (N=32) | 6 | 0.43 |
| Different length (N=64) | 5 | 0.33 |
| Different density (k=1) | 5 | 0.33 |
| Different density (k=4) | 4 | 0.25 |

**Heads present in all 7 conditions tested (the persistent core): layer 1/head 24, layer
3/head 24, layer 15/head 5.** 27 distinct heads appear across all 7 conditions' top-10 lists
(vs. 10 per condition), meaning roughly 2/3 of the "top-10" set is condition-specific while
a small stable core recurs regardless of seed, length, or density.

**Suggested placement**: Section 4.3, right after the induction/positional head taxonomy —
this is a natural companion finding ("what kind of heads are these" + "how stable is the
set").

---

## 5. Temperature / sampling robustness (answers: "are the trends robust to sampling, or is
everything greedy-decoding-only?")

**Why**: every other result in the entire project uses greedy decoding (`do_sample=False`).
This was an explicitly flagged limitation with no data behind it.

**Code**: `scripts/run_temperature_robustness_check.py` (new). Scoped as a targeted
robustness spot-check, not a full re-run under sampling: cyclic_shift/alphabet=2, N ∈
{16,32,64}, Base vs. the single densest headline condition (k=2), T ∈ {0.0, 0.2, 0.7},
top_p=0.95, 10 trials/cell.
**Data**: `results/Qwen2.5-7B-Instruct/temperature_robustness_grid.csv`

**Result — the effect survives sampling at both temperatures tested**:

| N | Delta at T=0.0 | Delta at T=0.2 | Delta at T=0.7 |
|---|---|---|---|
| 16 | +0.456 | +0.456 | +0.388 |
| 32 | +0.441 | +0.363 | +0.441 |
| 64 | +0.198 | +0.216 | +0.358 |

(Delta = Cellular_Bounds TAR − Base TAR.) The boundary benefit stays large and positive at
every length under both non-zero temperatures tested — the qualitative finding is not an
artifact of greedy decoding. TAR itself shows mild temperature-dependent variation as
expected from sampling noise; direction and rough magnitude of the effect do not change.

**Suggested placement**: Either a short new paragraph in Section 7 (Limitations) — moving
this from "uncharacterized" to "spot-checked, robust at T=0.2/0.7" — or its own small
appendix table if a fuller characterization is wanted later.

---

## 6. Benjamini-Hochberg FDR correction (answers: "did you consider FDR control or a
hierarchical testing framework?")

**Why**: the review's most standard statistical objection — hundreds of per-cell significance
tests reported via delta/SEM sigma bands, no correction described.

**Code**: `tasks/common/compute_fdr_correction.py` (new, CPU-only). Corrects within logical
families separately (mixing totally different analyses in one correction isn't statistically
meaningful) — the main 9-task grid (every task×model×length×axis_val cell, k=3 excluded,
same convention as everywhere else), the 5-task reset-variant grid, and the three
already-pooled tables (causal ablation, Gemini, CLRS).
**Data**: `results_clean/_fdr_correction/*.csv` (5 files, one per family)

**CORRECTION (2026-07-27): the original 707-test main-grid count was wrong — it silently
included two models never described anywhere in `main.tex`.** `select_canonical_files()`
globs every CSV sitting in each task's `grid/` directory, so two leftover exploratory runs
got swept into the family: **OLMo-2-1124-7B-Instruct** (a complete 9-task grid, 116 tests —
real data, just never written up) and **Qwen2.5-72B-Instruct** (only 2 of 9 tasks, 31 tests,
and its own plot filenames are tagged `_PRELIMINARY` — an abandoned exploratory run, not a
finished model). Neither model appears anywhere in Methods, Section 4.9, or Appendix G. Fixed
`compute_fdr_correction.py` to restrict the main-grid family to exactly the 5 models the paper
documents (Qwen2.5-7B/14B/32B, Mistral-7B-Instruct-v0.3, Phi-3.5-mini-instruct) and reran it.
The old "707 tests, 398 survive" numbers below are superseded — **use the corrected row.**

**Result — holds up well**:

| Family | n tests | Sig. at >2σ uncorrected | Survive BH FDR=0.05 |
|---|---|---|---|
| Main 9-task grid (corrected, 5 documented models only) | **560** | **362** | **350** |
| Reset-variant grid | 45 | 29 | 29 (no loss) |
| Causal ablation (pooled) | 10 | 8 | 8 (no loss) |
| Gemini (pooled) | 9 | 6 | 6 (no loss) |
| CLRS-Text (pooled) | 2 | 0 | 0 (already null) |

12 of 560 main-grid tests lose significance under correction; all are marginal edge cells
(e.g. N=256/Qwen2.5-14B cyclic_shift, N=64/Phi-3.5 running_sum_local) — nothing load-bearing.
The causal-ablation, Gemini, and reset-variant tables (the strongest individual claims) survive
correction completely intact.

**Systemic, not isolated**: the same bug (glob-based model discovery picking up leftover runs)
turned out to also be silently contaminating section 7 (dose-response robustness) and section
8 (completion rate) below — both were built on the same `select_canonical_files()`. Fixed once
at the source (`DOCUMENTED_MODELS` filter added directly inside `compute_diagnostics.py`'s
`select_canonical_files()`, defaulting on) and all three downstream scripts rerun; corrected
numbers are in each section below. Also **physically moved** every OLMo/Qwen2.5-72B grid CSV,
diagnostics-rows file, and the two `_PRELIMINARY`-tagged plots out of the active
`results_clean/<task>/grid/` directories into `results_clean/_excluded_undocumented_models/`
(nothing deleted, just quarantined — see the README there) so this can't recur silently again.

**Decision made (2026-07-27): both models stay out of the paper.**
- **OLMo-2-1124-7B-Instruct**: genuinely complete data, not a rough draft — full 9-task grid,
  same scope/trial-count parity as Mistral-7B-Instruct-v0.3 (identical row counts per task,
  40 trials/cell, up to N=256 for cyclic_shift). Ran 2026-07-19 (slurm jobs
  36215367-36216937). **Decided: leave it quarantined, do not add as a 6th model** — not worth
  the added Methods/Appendix scope and re-verification burden at this stage. No paper-text
  action needed; just don't let it leak back into any results aggregate (it's already moved to
  `results_clean/_excluded_undocumented_models/` and the FDR/dose-response/completion-rate
  scripts now exclude it by default at the source).
- **Qwen2.5-72B-Instruct**: genuinely incomplete — only 2 of 9 tasks (cyclic_shift,
  composed_permutation), and its own plot filenames were already self-labeled `_PRELIMINARY`.
  Ran 2026-07-16/17 (slurm jobs 36189429/36189473); cyclic_shift alone took 7h38m on 2×A100 —
  almost certainly a cost-driven stop, not a data problem. Stays quarantined, same as OLMo.

**Suggested placement**: A new paragraph in Section 2.5 (Statistical methodology) describing
the correction, plus a one-line note wherever the "560 tests, 350 survive BH" headline number
is most useful (likely the Discussion's methodological-cautions paragraph).

---

## 7. Dose-response model-selection robustness: AIC/BIC, leave-one-density-out CV, bootstrap
vertex CIs, vertex-vs-N regression (answers: "did you attempt cross-validated model selection
or information-criterion comparisons... can you report CIs for the vertex location?")

**Why**: the existing 45-row table compares mean R² only — a higher-order family always fits
better in-sample by construction, and the review specifically asked for complexity-penalized
and out-of-sample comparisons.

**Code**: `tasks/common/compute_dose_response_robustness.py` (new, CPU-only). Fits
linear/quadratic/saturating-exponential to every *individual* (task×model×length×axis_val)
curve (not just the pooled summary), computes AIC and BIC per curve, leave-one-
density-out CV (refit excluding each point, predict it, sum squared error — genuine held-out
prediction, not an in-sample fit statistic), and for quadratic fits, the vertex location with
a 500-iteration bootstrap CI. Also regresses vertex location on N per task/model (the
practitioner recipe for choosing k), both ungated and gated to only CV-validated curves.
**Data**: `results_clean/_dose_response_robustness/dose_response_robustness_full.csv` (348
rows), `vertex_vs_N_recipe.csv` (ungated, per task/model), `vertex_vs_N_recipe_cv_gated.csv`
(CV-gated, per task/model — new)

**CORRECTION (2026-07-27): row count and all numbers below were recomputed.** The original
441-row table included OLMo-2-1124-7B-Instruct and Qwen2.5-72B-Instruct curves — see section 6's
provenance note for the same root-cause bug (`select_canonical_files()` globbing in
undocumented models). Fixed at the source; corrected count is 348 curves (5 documented models
only). Directional conclusion is unchanged, numbers shifted slightly:

**Result — important, worth stating carefully in the text**:

| Metric | Quadratic wins | out of |
|---|---|---|
| AIC | 234 | 348 (67%) |
| BIC (stricter penalty) | 245 | 348 (70%) |
| Leave-one-density-out CV (true held-out) | **61** | **348 (17.5%)** |

Still ~17-18% under true held-out CV (was 74/441=17% before correction — same conclusion,
consistent magnitude). AIC/BIC (in-sample fit + an analytic complexity penalty) still favor
quadratic most of the time. But genuine out-of-sample prediction only favors it in roughly 1
of 6 curves at the individual-curve level — most curves have only 4-15 density points, not
enough to reliably distinguish a 2- vs. 3-parameter family from noise. **This does not
contradict the original pooled/mean R² claim** (which is about the aggregate across many
curves, and that's a different, still-valid statement) — but "quadratic robustly wins" needs
to be scoped explicitly: strong in aggregate, noisy per curve.

**Vertex-vs-N "practitioner recipe," now actually CV-gated (new this pass)**: applied the
CV-favors-quadratic gate the earlier version of this section flagged as still-needed. Result is
an honest negative: of 258 valid-vertex curves, only **55 survive the CV gate**, and once
grouped by (task, model) and required to have ≥4 points across ≥3 sequence lengths (the
minimum to fit a regression line at all), only **2 of 45 task/model combinations** have enough
CV-validated curves to produce a gated recipe — both `cyclic_shift`, both larger models
(Qwen2.5-14B: slope=-0.0008/token, intercept=0.765; Qwen2.5-32B: slope=-0.0033/token,
intercept=0.943; predicted vertex barely moves between N=8 and N=64 for either). **The
CV-gated "recipe" is not supportable as a general practitioner formula — there just isn't
enough validated per-curve signal outside of one task.** The ungated version (31 task/model
recipes, ready to use as a descriptive heuristic) is unchanged in spirit from before but
should not be framed as CV-validated.

**Suggested placement**: Section 3.2 (dose-response), as a new paragraph immediately after
the existing R²-comparison claim, explicitly distinguishing the aggregate vs. per-curve
robustness. For the Discussion's "Practical guidance" %TODO: **recommend dropping the
per-task/model vertex-vs-N formula idea** rather than filling it in with the ungated numbers —
the CV-gated result is that it isn't well-supported by the data. If a concrete recipe is still
wanted there, the only defensible one is the `cyclic_shift` pair above, explicitly scoped to
that task.

---

## 8. Completion rate as a function of boundary density (answers: "any evidence that
copy-through markers compound early termination?")

**Why**: cheap, existing-data question — computed from the already-stored diagnostic
row-level CSVs, broken out by `Interval` instead of pooled Base-vs-Bounds.

**Code**: `tasks/common/compute_completion_rate.py` (new — was inline/ad hoc before, now
persisted since it needed a rerun anyway). Reuses `results_clean/_diagnostics/*__rows.csv`.
**Data**: `results_clean/_diagnostics/completion_rate_by_density.csv`,
write-up in `results_clean/_diagnostics/Q7_completion_rate_writeup.md`

**CORRECTION (2026-07-27): recomputed — same OLMo/72B contamination as sections 6-7.** 11 of
the original 54 task×model row files were OLMo/72B; excluded now (moved to
`results_clean/_excluded_undocumented_models/diagnostics_rows/`). This one **changes the
qualitative claim, not just the numbers**: under the corrected 5-model set, the per-task
density↔completion-rate correlation is negative for **all 9 of 9 tasks** (was "7 of 9, with
`running_sum_local` reversing" before — that reversal was an OLMo artifact). Magnitudes also
dropped across the board (strongest is now `cyclic_shift` at -0.41, not `running_xor` at
-0.81). Full corrected tables are in the writeup file — use those, not any earlier copy of
this section's numbers.

**Result — real effect, task-dependent, not universal**:

Pooled across all tasks/models, completion rate is *not* simply monotonic in density (worst
at both extremes — sparsest k=16 at 0.661 and densest k=1 at 0.660 — best at moderate k=4,
0.786). But per-task correlation between density and completion rate is negative (the
expected "denser markers -> more early stopping" direction) in **7 of 9 tasks**, most strongly
for `running_xor` (corr=-0.808). `running_max` is roughly flat (-0.090) and `running_sum_local`
mildly reverses (+0.164).

**Suggested placement**: Section 3.8 (well-formedness tax), one or two sentences: the
mechanism is real and directionally dominant but not universal — state it in those terms
rather than as a clean confirmation.

---

## 9. Mean-replacement ablation + log-probability deltas (answers: "can you report log-prob
changes... and compare post-softmax masking to mean-replacement ablation to address
potential OOD concerns?")

**Why**: the existing causal ablation only ever zeroed target heads' attention to boundary
positions and measured TAR. This adds a gentler control (replace boundary-position attention
with that row's own mean attention weight, instead of zero+renormalize — tests whether it's
specifically the *absence* of boundary attention that matters, or any perturbation of the
normal pattern) and a more sensitive, continuous metric (teacher-forced log-probability of the
true output, not just binary correctness).

**Code**: `scripts/run_causal_ablation_meanreplace_logprob.py` (new). Reuses the exact target
heads already identified and saved in `causal_ablation_target_heads.csv` (not re-identified),
same task/model/density (cyclic_shift, alphabet=2, k=2) as the original ablation.
**Data**: `results/Qwen2.5-7B-Instruct/causal_ablation_meanreplace_logprob.csv`

**Result — zeroing causes far more disruption than mean-replacement, and the gap grows with length**:

| N | Δ log-prob, Mean-Replaced | Δ log-prob, Zero-Ablated |
|---|---|---|
| 8 | -0.037 | -0.118 |
| 16 | +0.007 (~flat) | -0.200 |
| 32 | -0.041 | -0.540 |
| 64 | -0.186 | **-0.593** (~3x the mean-replaced drop) |

Directly answers the OOD-shift concern: it's specifically the *removal* of boundary attention
that disrupts the model, not any deviation from its normal attention pattern — mean-replacement
(still an unusual attention pattern, just not zeroed) causes much less damage.

**Suggested placement**: Section 4.2, as a new paragraph right after the existing K-sweep/
matched-controls paragraph — this is a fourth control, stronger than the existing three
because it's a continuous (log-prob) rather than binary (TAR) measure.

---

## 10. Jitter placement (answers: "how sensitive is the effect to violating perfect
periodicity?")

**Why**: Appendix B already tests the *extreme* case (markers scattered uniformly at random,
ignoring periodicity entirely). The review specifically asked about small jitter, not full
scatter.

**Code**: `scripts/run_jitter_placement.py` (new). Canonical periodic positions {k,2k,3k,...}
each perturbed by +/-1 or +/-2 positions (drawn fresh per boundary, per trial), clipped to
stay in-range, duplicates removed (boundary count can drop very slightly from the exact-count
condition — logged per trial). Same task/model/density (cyclic_shift, alphabet=2, k=4) as the
other marker-property ablations, compared against the existing exact-periodic condition at the
same seeds.
**Data**: `results/Qwen2.5-7B-Instruct/jitter_placement_grid.csv`,
`results/Qwen2.5-7B-Instruct/jitter_placement_comparison.csv`

**Result — even small jitter mostly erases the benefit**:

| N | Base | Exact-periodic | Jitter ±1 | Jitter ±2 |
|---|---|---|---|---|
| 8 | 0.675 | 0.738 | 0.581 | 0.675 |
| 16 | 0.591 | 0.666 | 0.572 | 0.566 |
| 32 | 0.550 | 0.588 | 0.572 | 0.509 |
| 64 | 0.521 | 0.615 | 0.502 | 0.491 |

Exact-periodic clearly beats Base at every length. Jittered placement mostly does *not* —
at N=64 both jitter amounts fall *below* Base. The effect is far more sensitive to exact
regularity than "approximately periodic" — a sharper, more specific version of the existing
random-scatter finding in Appendix B.

**Suggested placement**: Appendix B, immediately after the existing "placement regularity is
essential" bullet — this quantifies exactly how strict "regular" needs to be.

---

## 11. External Tier-1/Tier-2 tasks sourced from real literature/benchmarks (answers: "have
you tried semi-natural tasks where the dependency is local," and generally strengthens
external validity beyond CLRS-Text/Gemini)

**Why**: CLRS-Text is the only non-invented external benchmark in the study so far, and both
its usable tasks (`minimum`, Kadane's) are full/one-step-dependency shapes, not position-local
(Tier-1). This adds genuine Tier-1 external checks, plus an attempted Tier-2 one.

**Selection process**: audited all 23 task types in HF dataset `ifujisawa/procbench`
(Fujisawa et al., arXiv:2410.03117) against three criteria — string type, length-preserving,
genuinely per-position — by inspecting real generated examples for each, not just prompt
text. Only **1 of 23 survived**: `task06` ("Substitute," a per-character table lookup,
verified exactly length-preserving and per-position with zero chaining across 6+ sampled
examples). Everything else failed on inspection: global operations (sort, rotate), list/
non-string outputs, length-changing operations, or — for one initially-promising candidate
(`task11`) — turned out to expand length on inspection despite looking per-character in the
prompt text.

**Code**:
- `tasks/common/procbench_vendor/generators.py` — vendored (MIT-licensed) ProcBench reference
  generator (`GeneratorSubstitute`), not a reimplementation, same principle as the CLRS-Text
  integration. Supports arbitrary length (native ProcBench data tops out at 25 characters,
  10 examples/length — not enough for this study's grid, so the real generator is called
  directly at our own lengths/seeds instead of using the fixed 240-example dataset).
- `tasks/procbench_substitute/engine.py` — the task engine.
- `tasks/string_copy/engine.py` — plain string copying per Jelassi et al. 2402.01032
  ("Repeat After Me"), alphabets 2/4/8, our own procedural generation (the paper's task
  definition, not external example data).
- `tasks/common/flip_flop_shared.py` + `tasks/flip_flop_high/engine.py` +
  `tasks/flip_flop_low/engine.py` — flip-flop language modeling per Liu et al. 2306.00946,
  reframed as per-position transduction (output the register-state stream at every position,
  not just at "read" query points, to fit this study's convention). High/low write-instruction
  density controls effective dependency length (frequent writes = short lookback; rare writes
  = long lookback).
- `tasks/common/resumable_runner.py` + `tasks/run_external_task.py` — new runner supporting
  incremental CSV append + resume (a killed/requeued job picks up where it left off), separate
  from the main study's `results_clean/` under `results/external_tasks/`.

All 4 engines hand-verified against real generated examples (traced the substitution table,
the identity copy, and the register-state recurrence by hand) before any GPU run.

**Screening (10-trial Base-only, chance-drop rule applied)**:
`results/external_tasks/SCREEN_RESULTS.md` has full detail.
- `string_copy`: survives, far above chance at every cell.
- `procbench_substitute`: survives, above chance at every length (N=4: TAR=0.350 vs.
  chance=0.029, down to N=64: TAR=0.053 vs. chance=0.029).
- `flip_flop_high`/`flip_flop_low`: **dropped, final** (attempted-and-unsuccessful rescue, not
  a silent omission). Original screen: below chance at every length, checked raw outputs before
  accepting this — not a scoring bug, the model frequently emits a single character (e.g. `"i"`
  or `"0"`) instead of the full N-character state stream, then stops. **Update (2026-07-27)**:
  attempted the fix — one narrow, targeted sentence added to the prompt explicitly stating the
  required output length and clarifying that 'r'/'i' are not a final query. Re-screened at
  Qwen2.5-7B-Instruct only, per an agreed stop-regardless-of-outcome plan (no 32B escalation, no
  further wording iteration either way). **Still below chance at every length after the fix**
  (flip_flop_high: 0.34-0.44 TAR vs. 0.5 chance; flip_flop_low: 0.13-0.36 TAR vs. 0.5 chance) —
  same completion failure persists in the raw outputs. Full detail in
  `results/external_tasks/SCREEN_RESULTS.md`. Report this as a real, documented negative result
  (external Tier-2 check attempted, prompt-clarity hypothesis tested and did not resolve it),
  not as an open item.

**Full grid (20 trials, matching the main study's N/k grid, on the 2 survivors)**:

`string_copy` — Base is already near-ceiling at every alphabet (0.97-0.98 TAR): essentially
nothing for boundaries to scaffold. Deltas are tiny (+0.002 to +0.007) and non-significant
(sigma 0.14-0.61). A clean, informative null — the same "nothing to scaffold at ceiling" story
as the main study, at the most extreme end.

`procbench_substitute` — Base TAR degrades with length as expected (0.53 at N=4 down to 0.06
at N=64). Boundaries show a **positive delta at every single length tested** (5/5): overall
pooled delta +0.044, sigma=1.64 (short of 2sigma on the pooled number, but 5-for-5 same-
direction is itself informative -- sign test p~=0.03). A second, independently-sourced Tier-1
external-benchmark result in the same direction as the main study's Tier-1 findings.

**Suggested placement**: A new subsection under 3.10 (External validity), alongside CLRS-Text
and Gemini — this genuinely closes the "only CLRS-Text, and it's not even Tier-1" gap. Mention
`flip_flop` as an attempted external Tier-2 check that failed a completion-behavior screen even
after a targeted prompt fix (honest, not hidden) if space allows, or in a footnote/limitations
sentence — this is now a closed, final result, not an open item.

---

## 12. k=3 sensitivity check for FDR + dose-response robustness (answers: "does excluding
Interval=3 throughout the study change any headline conclusion?")

**Why**: every FDR and dose-response result in this study excludes k=3 by convention (it most
often sits on a task's alphabet/modulus period, producing degenerate boundary placements for
some axis values) — worth confirming that convention isn't quietly propping up a conclusion.

**Code**: `tasks/common/compute_k3_sensitivity.py` (new, CPU-only). Reruns the FDR correction
(main_grid, reset_variant_grid families) and the dose-response robustness fits with k=3
INCLUDED, side by side with the standard (excluded) numbers.
**Data**: `results_clean/_k3_sensitivity/*.csv` (6 files: excluded/included pairs for
main_grid, reset_variant_grid, and the dose-response curve fits)

**Result — no material reversal of any headline claim; some sensitivity in one place**:

| Metric | k=3 excluded (standard) | k=3 included |
|---|---|---|
| FDR main_grid: n tests | 560 | 560 |
| FDR main_grid: BH-sig (FDR=0.05) | 350 (62.5%) | 330 (58.9%) |
| FDR reset_variant: n tests | 45 | 45 |
| FDR reset_variant: BH-sig (FDR=0.05) | 29 (64.4%) | 30 (66.7%) |
| Dose-response: AIC favors quadratic | 234/348 (67.2%) | 321/454 (70.7%) |
| Dose-response: BIC favors quadratic | 245/348 (70.4%) | 326/454 (71.8%) |
| Dose-response: CV favors quadratic | 61/348 (17.5%) | 100/454 (22.0%) |
| Dose-response: interior vertex | 258/348 (74.1%) | 316/454 (69.6%) |

Every metric moves by a few percentage points in one direction or the other, none reverses
sign or changes which side of "majority/minority" it's on. The largest single shift is FDR
main_grid's BH-survival count (350→330, a 3.6-percentage-point drop) — worth a one-line
mention if the Methods section states the k=3 exclusion, but not a result that needs
re-litigating.

**Suggested placement**: One sentence in the Methods paragraph that already states the k=3
exclusion convention (Section 2.5), citing this as a robustness check.

---

## 13. Alphabet tokenization audit (extends the existing marker-tokenization check;
answers: "does every alphabet character used across the 9 tasks tokenize as exactly one
token, on all 5 documented models?")

**Why**: `results/marker_tokenization_cross_model.csv` already checks that the boundary
marker `|` (and a few alternatives) is single-token on 4 models — this extends the same
question to the actual task alphabets (letters A-H, digits 0-7) on all 5 documented models,
since every mechanistic/attention analysis implicitly assumes one character = one token at
one sequence position.

**Code**: `tasks/common/compute_alphabet_tokenization_audit.py` (new, CPU-only,
tokenizer-only — no model weights loaded). Method note: measuring token count via a naive
isolated `tokenizer.encode(char)` call is unreliable for two reasons, both confirmed by
inspecting raw token IDs — (1) SentencePiece tokenizers (Mistral, Phi-3.5) inject a spurious
leading space-token when a string has no preceding context, which doesn't happen inside a
real prompt; (2) BPE merge behavior is context-dependent, so a single isolated character
doesn't necessarily tokenize the way it does surrounded by its real neighbors. Fixed by
diffing `tokens("Input: " + sequence)` against `tokens("Input: ")` — realistic prefix,
context-correct merge behavior, artifact-free.
**Data**: `results_clean/_diagnostics/alphabet_tokenization_audit.csv`

**Result — digits are clean; letters are NOT, on every one of the 5 documented models. This
is a real finding, not a minor audit note, and needs a Methods/Limitations mention**:

| Alphabet | Bare (k=0) tokens/chars ratio | Affected tasks |
|---|---|---|
| digits 0-1, 0-3, 0-7 | **1.00 on all 5 models** | running_max, running_sum_local, copy_previous, running_xor (4 of 9 tasks) — clean |
| letters A-B (size 2) | 0.33 (Qwen) / 0.58 (Mistral, Phi-3.5) | cyclic_shift, character_substitution, random_permutation, conditional_shift, composed_permutation (5 of 9 tasks) |
| letters A-D (size 4) | 0.47 (all 5 models) | same 5 tasks |
| letters A-H (size 8) | 0.40 (Qwen) / 0.54-0.58 (Mistral, Phi-3.5) | same 5 tasks |

At **Base (k=0, no boundaries)**, consecutive same-case letters routinely merge into
multi-character tokens — confirmed by direct inspection, e.g. Qwen's tokenizer contains
`'ABCDEFG'` as one literal single token; a 48-character A-H test string tokenizes to just 19
tokens on Qwen (ratio 0.40), 26-28 on Mistral/Phi-3.5. **Boundary insertion breaks this up,
but only mostly**: at k=1 (a boundary after every single character) the sequence becomes
essentially bijective again (off by exactly 1 token — a negligible prefix-seam effect, same
magnitude on every model/alphabet). At k=2, 4, 8, and 16 — the actual densities used
throughout the grid — substantial merging persists inside each between-boundary segment (e.g.
at k=4, Qwen's A-H string is still missing 22 of 67 expected tokens).

**Why this matters for the paper's central claim**: for the 5 letter-alphabet tasks, the
**Base condition's input is silently harder to tokenize** than the **Cellular_Bounds
condition** — the model has to internally decompose multi-character tokens it never has to
face once boundaries are inserted densely enough. This is a plausible partial confound for
"boundaries help": some of the measured improvement could be a tokenization-scaffolding
effect (cleaner, more atomic input representation) rather than purely an attention-mechanism
effect. It does not invalidate the finding — the digit-alphabet tasks (running_xor,
running_max, etc., which show the same qualitative pattern) have zero tokenization
confound and still show boundaries helping in some cases and hurting in others, so
tokenization can't be the sole explanation — but it is a real, previously-undocumented
confound specific to the letter-alphabet tasks that a careful reviewer could raise, and it's
better to disclose and discuss it than have it discovered.

**Suggested placement**: This needs an actual paragraph, not a one-liner — recommend a new
short subsection in Limitations (or Section 2.5 Methods, near the tokenization/marker
discussion that already exists) explicitly noting: digit-alphabet tasks have no tokenization
confound; letter-alphabet tasks do, it's largest at low boundary density, and boundary
insertion itself mechanically reduces it (a second, non-attention channel through which
denser boundaries could plausibly help letter-alphabet tasks specifically). Cite this
section's table.

---

## 14. Reset-variant second batch — now complete, §4.5 goes from single-batch to verified

**Why**: the 5 causal-ablation reset-variant tasks (running_xor_reset, copy_previous_reset,
running_sum_local_reset, reverse_xor_reset, running_max_reset) had only ever been run as a
single 40-trial batch — everything else in the study gets a second, independent,
non-overlapping batch as a replication check. This closes that gap.

**Code**: existing `tasks/run_task.py`/`tasks/common/runner.py` `trial_start` mechanism
(already used for the main study's second batches), run via `tasks/run_task.sbatch` — a
genuine infrastructure bug was found and fixed in this sbatch script mid-session (it wasn't
forwarding the 5th CLI arg, so the first submission attempt silently reran trials 0-39 instead
of 40-79 for all 5 jobs; caught before any data was corrupted, fixed, and resubmitted
correctly — see the earlier bugfix note if this file's history matters to you).
**Data**: `results_clean/<task>_reset/grid/Qwen2.5-7B-Instruct_trials40-79.csv` (5 files, all
verified to cover exactly Trial 40-79, non-overlapping with the original Trial 0-39 batch)

**Result — replicates cleanly; every qualitative claim holds up, `running_max_reset` is the
standout**:

| Task | Batch 1 (n=40) Base/Bnd | Batch 2 (n=40) Base/Bnd | Pooled (n=80) Delta | Pooled σ |
|---|---|---|---|---|
| running_xor_reset | 0.448 / 0.492 | 0.482 / 0.493 | +0.016 | 1.5 |
| copy_previous_reset | 0.598 / 0.556 | 0.576 / 0.568 | −0.036 | −3.3 |
| running_sum_local_reset | 0.292 / 0.352 | 0.288 / 0.347 | +0.057 | **9.4** |
| reverse_xor_reset | 0.467 / 0.524 | 0.498 / 0.512 | +0.029 | 2.8 |
| running_max_reset | 0.700 / 0.472 | 0.694 / 0.469 | **−0.199** | **−20.8** |

Batch 1 and batch 2 are consistent with each other in every case (same sign, similar
magnitude, no case where the second batch reverses the first) — this is exactly what a
genuine replication should look like, not a fluke single run. `running_max_reset` shows by
far the largest, most robust effect in either direction (boundaries hurt this task
substantially, σ≈−21, essentially unchanged between batches: −0.228 batch 1 vs. −0.225 batch
2). `copy_previous_reset` is the only one where boundaries hurt rather than help, consistent
across both batches. `running_sum_local_reset` and `reverse_xor_reset` both show boundaries
helping, consistent across both batches.

**Suggested placement**: Section 4.5 — replace "single 40-trial batch" language with "two
independent 40-trial batches (n=80 pooled), consistent across both" and cite the pooled
deltas/σ above. The corresponding Limitations bullet about the reset-variant tasks only
having single-batch verification should be deleted. Appendix G's per-task matrix should add
the pooled n=80 numbers in place of (or alongside) the original n=40 ones.

---

## 15. Gemini per-length x per-density breakdown tables (appendix-ready)

**Why**: the existing Gemini section only reports 9 pooled (Task, Base-vs-Bounds) numbers —
this expands each task to the full Seq_Length x Interval grid (mean TAR, SEM, n) matching the
level of detail already given for the 5 documented open-weight models.

**Code**: `tasks/common/compute_gemini_breakdowns.py` (new, CPU-only, reuses the existing
pooled 3-trial Gemini CSV — no new API calls).
**Data**: `results_clean/_gemini_breakdowns/<task>_breakdown.csv` (9 files) and
`results_clean/_gemini_breakdowns/gemini_breakdowns_appendix.md` (all 9 tables, already
formatted as markdown tables ready to paste into an appendix).

**Suggested placement**: A new appendix table (or extension of the existing Gemini appendix
table if one exists) — paste `gemini_breakdowns_appendix.md`'s content directly, or point the
paper-writing session at the per-task CSVs to reformat.

---

## 16. WFR robustness under near-greedy sampling (does the negative WFR delta survive
temperature > 0?)

**Why**: extends `scripts/run_temperature_robustness_check.py` (which only reported TAR/SAR
under sampling) to the full well-formedness diagnostic suite. Question: the study elsewhere
finds cyclic_shift's Cellular_Bounds condition is well-formed *less* often than Base at greedy
decoding (a negative WFR delta) even though TAR strongly improves — is that a greedy-decoding
artifact, or does it survive real sampling?

**Code**: `scripts/run_wfr_decoding_robustness.py` (new, GPU). Same grid as the original
temperature check (cyclic_shift, alphabet=2, N∈{16,32,64}, Base vs. k=2, 10 trials/cell,
T∈{0.0, 0.2, 0.7}), but now saves the full untruncated `Raw_Model_Output` and computes
Well_Formed/Vocab_Valid/Complete via `tasks.common.metrics.row_diagnostics` — the exact same
diagnostic definitions used everywhere else in the study, not a reimplementation.
**Data**: `results/Qwen2.5-7B-Instruct/wfr_decoding_robustness/wfr_decoding_robustness_grid.csv`
(180 rows), `wfr_decoding_robustness_summary.csv`

**Result — the negative WFR delta is NOT a greedy-decoding artifact; it persists, and if
anything grows slightly under sampling, while TAR's improvement also persists**:

| Temperature | TAR Base | TAR Bounds | WFR Base | WFR Bounds | WFR Delta |
|---|---|---|---|---|---|
| 0.0 (greedy) | 0.514 | 0.879 | 0.833 | 0.800 | −0.033 |
| 0.2 | 0.536 | 0.881 | 0.833 | 0.767 | −0.067 |
| 0.7 | 0.508 | 0.904 | 0.867 | 0.800 | −0.067 |

VVR (Vocab_Valid_Rate) is 1.0 at every temperature/method — the model never emits an
out-of-alphabet character regardless of sampling. So the negative WFR delta is entirely a
Complete_Rate effect (Cellular_Bounds stops/truncates slightly more often than Base), and both
that effect and TAR's large positive delta hold up cleanly across greedy and near-greedy
sampling — this specific finding is not an artifact of do_sample=False.

**Suggested placement**: One sentence in Section 4.8 (wherever the greedy-decoding WFR finding
is currently reported), citing this table as confirming it survives T=0.2/0.7 sampling.

---

## 17. Anchor attention geometry: does the boundary-attention mechanism look different when
boundaries hurt instead of help? (copy_previous vs. cyclic_shift)

**Why**: every existing attention/mechanistic analysis in the paper is on cyclic_shift, a task
where boundaries help. `copy_previous` is a Tier-2 anchor where boundaries measurably *hurt*
(reset-variant section above: pooled delta −0.036, σ≈−3.3 for the reset variant; the original
non-reset `copy_previous` also shows boundaries hurting). Question: does the same geometric
signature (boundary tokens accumulating attention mass as density increases, entropy/diagonal/
distance shifts) fire identically regardless of whether it helps or hurts the task, or does it
look different specifically when boundaries are harmful?

**Code**: `scripts/run_anchor_attention_geometry.py` (new, GPU). Same correlational attention
battery, matched configuration and seeds, as `scripts/run_attention_templated.py`
(cyclic_shift's headline attention sweep): N∈{4,8,16,32,64,128,256}, k∈{2,4,8} (k<N only), 20
trials/cell, same metrics (boundary-token attention mass, entropy, off-diagonal/distance,
clustering), same local-seed formula. One real bug found and fixed before this ran cleanly:
copy_previous's prompt (unlike cyclic_shift's) contains an "Example: Input 1011 -> Output
1101" line *before* the real "Input: ...\nOutput:" line — the original crop logic (copied from
the cyclic_shift script, which has no such example line) found the *first* "Input"/"Output"
occurrence and silently cropped the wrong token span entirely. Fixed by taking the *last*
occurrence instead (verified against a raw tokenization trace before rerunning).
**Data**: `results/Qwen2.5-7B-Instruct/anchor_attention_geometry/attention_metrics_copy_previous.csv`
(360 rows), `attention_run_0_copy_previous.npz` (full attention tensors, representative
N=16/k=2 example)

**Result — the mechanism fires the same way in both directions; magnitude differs, sign
doesn't. The attention-geometry signature alone does not predict whether boundaries help or
hurt.**

Boundary-token attention mass vs. density (1/k), linear regression by length — same strong,
clean linear relationship on both tasks (R²=0.98-1.00 at every length, both tasks):

| N | cyclic_shift slope (helps) | copy_previous slope (hurts) | ratio |
|---|---|---|---|
| 8 | 0.214 | 0.158 | 0.74 |
| 16 | 0.246 | 0.173 | 0.70 |
| 32 | 0.231 | 0.159 | 0.69 |
| 64 | 0.214 | 0.153 | 0.71 |
| 128 | 0.207 | 0.150 | 0.72 |
| 256 | 0.203 | 0.148 | 0.73 |

Pooled delta metrics (Cellular_Bounds − Base, averaged across the full sweep):

| Metric | cyclic_shift (helps) | copy_previous (hurts) |
|---|---|---|
| Delta_Entropy | +0.120 | +0.104 |
| Delta_Diag | +0.0009 | +0.0018 |
| Delta_Dist | +0.0036 | +0.0045 |
| Delta_Clustering | +0.0011 | +0.0081 |

**Same direction on every single metric, both tasks** — boundary tokens draw attention mass
that scales linearly with density, attention entropy rises, diagonal (self-)attention and
attention-weighted distance both increase slightly, in both a task where this measurably helps
and one where it measurably hurts. The magnitude is consistently *weaker* on copy_previous for
the headline boundary-attention-mass slope (69-74% of cyclic_shift's, remarkably stable across
all six lengths) — boundaries pull somewhat less attention toward themselves on the task where
they're harmful, which is at least directionally consistent with "less scaffolding effect where
scaffolding isn't needed/wanted." Clustering is the one metric where copy_previous's delta is
larger in absolute terms (7x cyclic_shift's, though both are small, <0.01) — worth a mention if
Section 5.1 gets into clustering specifically, but not the headline point.

**Bottom line for the paper's mechanistic claim**: this is genuine evidence *against* "boundary
tokens mechanically capturing attention mass is *why* boundaries help" as a sufficient
explanation on its own — the same capture happens, in the same functional form, on a task where
it doesn't help at all. Whatever determines the direction (help vs. hurt) has to come from how
the *rest* of the attention pattern uses that reallocated mass relative to what the task
actually needs (copy_previous needs attention on the immediately-preceding token; cyclic_shift's
computation is more local-symbol-mapping and less disrupted by a redirected step) — not from
whether the redirection happens at all.

**Suggested placement**: A new paragraph in Section 5.1, directly closing the "mechanistic
generality beyond Tier-1 / does this replicate on a task where boundaries hurt" gap. This is a
substantive qualifying finding for the paper's causal story, not just a confirmation — flag it
as such rather than folding it in as a minor footnote.

---

## 18. Causal head ablation on copy_previous (answers: "does ablating boundary-attending heads
mitigate the harm on a task where boundaries hurt?" — the causal completion of the anchor-
attention-geometry correlational finding)

**Why**: Section 4.1's anchor-geometry result showed the boundary-attention mechanism fires
identically on copy_previous (harm) and cyclic_shift (help) — correlationally. This tests
whether it's also causally load-bearing for the *harm*, the same way it's load-bearing for
the benefit on cyclic_shift (Section 4.2). Either outcome is informative: harm mitigated ⇒
the same circuit carries both; harm unchanged ⇒ disruption routes through a different
pathway.

**Code**: `scripts/run_causal_ablation_copy_previous.py` (new, GPU). Same pipeline as
`scripts/run_causal_ablation.py` (fresh 5-trial diagnostic → top-10 boundary-attending heads
→ zero-and-renormalize those heads' boundary attention during generation), applied to
copy_previous at k=2, matched to the study's canonical seed formula
(`90_000*N + 100*k + 10*axis_val + t`) so the diagnostic pass, ablated run, and the paired
Base/normal-Bounds reference (read from the existing pooled 40-trial grid, first 20 trials)
all use identical sequences. One real bug fixed before running: the original script's boundary-
token locator takes the *first* "Input"/"Output" token occurrence, which is safe for
cyclic_shift's prompt but wrong for copy_previous's (which has an "Example: Input 1011 ->
Output 1101" line before the real one) — the same crop bug found and fixed in
`run_anchor_attention_geometry.py`, fixed here the same way (take the *last* occurrence).
**Data**: `results/Qwen2.5-7B-Instruct/causal_ablation_copy_previous/` (target heads,
full grid, Base/normal/ablated comparison)

**Result — the same circuit is causally implicated, but explains only a minority of the harm,
growing with length. Not a clean "mitigated" or "unchanged" — a third, more precise answer**:

- **Full 3/3 overlap** between copy_previous's top-10 boundary-attending heads and
  cyclic_shift's persistent 3-head core: `(1,24)`, `(3,24)`, `(15,5)` are all present, and rank
  among copy_previous's own top-3 by boundary-attention mass (0.072/0.060/0.052 vs. cyclic_
  shift's own ranking). The same small circuit attends to boundaries regardless of whether
  that attention helps or hurts the task.
- Ablating that circuit **partially mitigates the harm**, and the mitigated fraction grows
  with length, mirroring the shape (not the magnitude) of cyclic_shift's own ablation result:

| N | Base TAR | Normal Bounds TAR | Ablated Bounds TAR | Harm (normal) | Harm (ablated) | % Harm mitigated |
|---|---|---|---|---|---|---|
| 8 | 0.600 | 0.400 | 0.400 | −0.200 | −0.200 | 0.0% |
| 16 | 0.506 | 0.347 | 0.363 | −0.159 | −0.144 | 9.8% |
| 32 | 0.464 | 0.314 | 0.328 | −0.150 | −0.136 | 9.4% |
| 64 | 0.489 | 0.347 | 0.376 | −0.142 | −0.113 | 20.3% |

Compare to cyclic_shift's ablation erasing up to 62.6% of the *benefit* at N=64 — the same
circuit is doing real causal work in both directions, but it accounts for a much smaller share
of the effect on copy_previous, and what it does account for grows with length just like the
help-side effect does. This sharpens the mechanism section: the sink circuit is necessary-ish
for both directions but not sufficient to explain either one fully — most of copy_previous's
harm routes through something the top-10-head ablation doesn't touch.

**Suggested placement**: A new paragraph in Section 4.2, immediately after the existing causal
ablation result, as the causal completion of Section 4.1's anchor-geometry correlational
finding. Also worth a sentence in Section 4.3 (head taxonomy) noting the 3-head core is
shared across a help-task and a harm-task, not task-specific.

---

## 19. Digit-alphabet cyclic_shift (answers: "quantify how much of the Tier-1 letter-alphabet
benefit is tokenization-atomicity vs. attention-scaffolding")

**Why**: Section 2.2's disclosed tokenization confound (letter alphabets merge into multi-
character tokens at Base; digit alphabets don't) is currently qualitative. This makes it
quantitative: rerun cyclic_shift's identical rule/grid/seeds with digit alphabets (0-1, 0-3,
0-7) instead of letters (A-B, A-D, A-H) — since the local-seed formula only depends on
(N, k, axis_val, t), not on the alphabet's symbols, the same `rng.choices()` call draws
positionally identical sequences, just relabeled digit-for-letter. The letter-vs-digit
boundary benefit gap at matched cells directly measures the atomicity contribution.

**Code**: `tasks/cyclic_shift_digits/engine.py` (new task, registered in `tasks/run_task.py`)
— identical to `tasks/cyclic_shift/engine.py` except the alphabet is `[str(i) for i in
range(alphabet_size)]` instead of letters. Verified positionally matched to the letter
version by hand before running (same rng draws, symbols relabeled). Full grid run via the
existing `tasks/run_task.sbatch` pipeline (Qwen2.5-7B, 20 trials, same N/k grid as the
headline cyclic_shift study).
**Data**: `results_clean/cyclic_shift_digits/grid/Qwen2.5-7B-Instruct.csv`

**Result — digits still benefit from boundaries, confirming attention-scaffolding is real and
not purely a tokenization artifact — but roughly 2.4x less than letters, confirming the
confound has real magnitude, not just direction**:

| | Mean Bounds-Base gap (TAR), pooled across N/k/alphabet_size |
|---|---|
| Digit alphabet (this job) | **+0.061** |
| Letter alphabet (existing headline data, same 20 trials/seeds) | **+0.147** |

Digit-alphabet boundaries help in 15 of 21 (N, k, alphabet_size) cells (positive gap), the
same qualitative direction as letters, just weaker — confirming the core Tier-1 mechanism is
not solely a tokenization effect (the confound-free digit tasks elsewhere in the paper already
established this at the task-suite level; this is the first same-task, same-rule, alphabet-
only comparison). Cross-referenced against the existing `digit_complement` task (Section 3.6,
also a digit alphabet, different rule): its own k=2 gap averages +0.033 across N=8-64 —
consistent in magnitude with this job's +0.061 digit-cyclic_shift gap, both well below the
letter-alphabet +0.147. Taken together: roughly 40-60% of the letter-alphabet Tier-1 benefit
survives with tokenization confound-free digits (i.e. is genuine attention-scaffolding), and
the remainder (comparable in magnitude to the confound-free portion, not a minor correction)
is attributable to the atomicity gap Section 2.2 discloses.

**Suggested placement**: A new sentence or short paragraph directly in Section 2.2's
tokenization-confound paragraph, upgrading "a real partial confound for the magnitude" to a
quantified bound (roughly half). Could also be a one-line cross-reference from Section 3.6's
digit_complement discussion.

---

## 20. State-carrying running_max (answers: "quantify how much explicit state offsets
dependency disruption on running_max" — the remaining state-carrying data point alongside
modsum and addition)

**Why**: modsum and addition both already have a state-carrying redesign result (Section 3.7,
Appendix C); running_max — the easier, high-Base-competence Tier-3 anchor — was the remaining
piece of that story.

**Code**: `scripts/run_running_max_state_carrying.py` (new, GPU). Same design as
`scripts/run_modsum_state_carrying.py`: instead of a content-free `|` boundary, each boundary
is replaced by the TRUE running maximum at that point, written as `[v]` — a genuine checkpoint
rather than a divider the model is told to disregard. Verified by hand against a manual trace
before running. Only the new state-carrying condition is run fresh; Base and the original
content-free Cellular_Bounds (k=4) are read from the existing pooled 40-trial grid at
identical seeds (tasks.common.runner's local-seed formula) — a paired three-way comparison,
first 20 trials, alphabet_size ∈ {2,4,8}, N ∈ {8,16,32,64}.
**Data**: `results/Qwen2.5-7B-Instruct/running_max_state_carrying/`

**Result — checkpointing helps in proportion to how much state there is to lose, exactly
matching the modsum finding, and Base still dominates both, exactly matching modsum/addition**:

| Alphabet_Size | Mean (StateCarrying − ContentFree) | Mean (StateCarrying − Base) |
|---|---|---|
| 2 | −0.059 | −0.030 |
| 4 | +0.064 | −0.059 |
| 8 | +0.108 | −0.008 |

State-carrying beats the content-free marker in 7 of 12 cells overall, but the pattern is not
noise: the benefit rises monotonically with alphabet size (more possible state values to lose
track of ⇒ more benefit from an explicit checkpoint), the same "helps in proportion to how
much state there is to lose" conclusion Section 3.7 already draws for modsum. Base still beats
state-carrying in 8 of 12 cells (mean gap −0.032), but the gap is much smaller than modsum's
or addition's own state-carrying-vs-Base gaps — running_max's Base competence is already high,
leaving less room for any intervention, checkpoint included, to close the distance.

**Suggested placement**: A new sentence in Section 3.7, extending the modsum state-carrying
paragraph with running_max as a third, confirming data point (alongside modsum and addition's
existing state-carrying results) — completes the state-carrying story across all three
Tier-3/global-state tasks in the study.

---

## 21. Reset-instruction compliance audit — IMPORTANT, read before anything else in this
section (answers Q5: "does the model actually follow reset semantics at the position after
each boundary, or does it silently carry over the original computation?")

**Why**: Section 3.5's entire causal story rests on the reset variant actually being a
different task the model attempts differently, not just the same behavior re-scored against
different ground truth. Direct precedent for this kind of audit: Appendix B already found the
model disobeys a "do not echo" instruction in 90-100% of trials on a different task — this
asks the analogous compliance question for "reset at this boundary."

**Code**: `scripts/audit_reset_instruction_compliance.py` (new, CPU-only, no GPU needed —
uses already-stored `Input_Given`/`Ground_Truth`/`LLM_Output` from the existing reset-variant
grid CSVs). Method: at every position immediately after a boundary (before, for
`reverse_xor_reset`, whose dependency looks forward), recompute what the ORIGINAL
(non-reset, continuous) rule would predict there, using the real non-reset task's own
compute function (`tasks.running_xor.engine._compute_xor_chain`, etc. — not a
reimplementation). Only positions where the reset prediction and the original/carry-over
prediction actually *differ* are counted (48,224 of them, across all 5 tasks) — a position
where both predictions agree can't distinguish compliance from coincidence. Compared against
what the model actually output at that position. Verified by hand against a real example
(digits `6620454162705220`(a real trial's input): continuous carry-over predicts
`6666666666777777`; reset predicts `6666455566775555`; the model produced
`6666455567775555` — clean reset compliance in that specific case, confirming the audit logic
is measuring what it claims to).
**Data**: `results_clean/_reset_compliance_audit/reset_compliance_audit_rows.csv` (48,224
rows), `reset_compliance_summary.csv`

**Result — compliance is real but weak-to-moderate for most tasks, and running_max_reset is a
genuine concern that specifically threatens the "5x more harm" full-prefix causal claim**:

| Task | Follows RESET | Follows ORIGINAL carry-over | Follows neither | N distinguishing positions |
|---|---|---|---|---|
| reverse_xor_reset | 0.475 | 0.444 | 0.018 | 4,295 |
| running_xor_reset | 0.463 | 0.328 | 0.068 | 4,271 |
| copy_previous_reset | 0.402 | 0.364 | 0.188 | 4,214 |
| running_sum_local_reset | 0.385 | 0.159 | 0.368 | 18,388 |
| **running_max_reset** | **0.157** | **0.484** | 0.216 | 17,056 |

Four of five tasks show reset semantics modestly favored over carry-over (close to a coin
flip for `reverse_xor_reset`/`running_xor_reset`, more favored for `copy_previous_reset` and
`running_sum_local_reset`) — real but not strong compliance, consistent with the model
partially, not fully, internalizing the reset instruction. **`running_max_reset` is the
outlier and the concerning case**: the model follows the *original, carry-over* behavior over
3x more often than it follows reset semantics, and this gets substantially worse with length
(Follows_Reset drops monotonically 0.293→0.206→0.207→0.192→0.123 at N=4/8/16/32/64, while
Follows_Orig_CarryOver rises 0.366→0.406→0.398→0.464→0.519 over the same range). This is
specifically the task carrying Section 3.5's largest, most dramatic reset-ablation result
(harm growing ~5x, 2.87σ→14.91σ, described as the cleanest evidence that "chunking a
full-prefix dependency creates more, shorter copies of it") — and it's now the task with the
weakest evidence the model is actually attempting the reset task rather than just continuing
the original computation and being scored against different ground truth.

**What this does and doesn't mean**: it does NOT mean Table 2's numbers are wrong — TAR is
correctly computed against the stated reset ground truth regardless of what the model actually
does. It DOES mean the *causal interpretation* — "the harm grows because chunking multiplies
a dependency that can't be severed" — is harder to cleanly attribute for `running_max_reset`
specifically, since a model that's largely still doing the *original* task (not attempting a
reset) would show *increased* apparent "harm" almost mechanically (any chunk after the first
inherits an increasingly stale/wrong running-max baseline relative to reset ground truth,
independent of whether "dependency chunking" as a mechanism is real). The qualitative
direction of the finding (full-prefix tasks fare worse under chunking than one-step tasks)
may still hold, but the specific "~5x" magnitude and the mechanistic story built on it should
be qualified by this compliance gap rather than presented as clean causal evidence. The four
one-step reset tasks are on firmer ground (real, if imperfect, compliance).

**Suggested placement**: This needs a real paragraph in Limitations, and probably a qualifying
sentence directly in Section 3.5 next to the running_max_reset result — not just a footnote.
Recommend explicitly softening the "5x more harm... whose dependency chunking can only
subdivide, never remove" framing to note the compliance caveat, since a careful reviewer who
thinks to ask "did the model actually attempt the reset task" (exactly this question) will
otherwise catch it.

---

## Two factual discrepancies found while independently verifying the current draft (please
fix in the text, not just note them)

1. **Section 4.3**: "nine of ten rank below the population median induction score... with
   one striking exception" is not quite right. The actual data
   (`results/Qwen2.5-7B-Instruct/induction_vs_boundary_heads.csv`) shows **8 of 10** below
   median, with **two** heads above it: layer 22/head 5 (97.2nd percentile, the one
   mentioned) and layer 21/head 6 (70.5th percentile, not mentioned). The mean (30.9) is
   correctly stated and doesn't need to change. Fix: "eight of ten... two exceptions" (or
   drop the "one exception" framing and just report the mean + list both outliers).

2. **Appendix C**: "Base tar exceeds every single tested boundary density... at all 20
   (model, length) combinations" is off by one cell. Checked directly against
   `addition_lsd_grid_templated_v2.csv` for all four models — 19 of 20 combinations are
   clean Base wins; **Qwen2.5-32B at N=8 is an exact tie** (Base 0.475 = best-density
   0.475), not a Base win. Fix: "19 of 20... with one exact tie at Qwen2.5-32B, N=8" (the
   qualitative claim — reversing digit order does not rescue Cellular — still holds; this is
   a precision fix, not a retraction).
