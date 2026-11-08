# Paper plan — boundary markers (status as of 2026-11-07)

This replaces `NEXT_STEPS.md` and `REVIEWER_RESPONSE_PLAN.md` (July, out of date).

**Source of truth for results:** `boundaries_report_iclr.html` (curated figures and tables, Qwen2.5-32B primary).
`boundaries_report.html` is the full findings report behind it.

**Do not write from `iclr_submission.pdf`.** It was built on 2026-07-29, before the August rework, and several of its
headline claims no longer hold (see "Claims to retire" below). Its LaTeX source is not in this repo
(`overleaf/main.tex` is older, Jul 13), so it is probably on Overleaf.

---

## 1. Framing

### The problem with the current headline

"Boundaries help position-local tasks and hurt dependency tasks" is true at 7B/14B, Mistral, Phi and Gemini,
but **not at Qwen2.5-32B, the paper's primary model** (Table 6, pooled TAR change, Base → Cellular_Bounds):

| 32B, pooled | Original Δ | Reset Δ (dependency severed) |
|---|---|---|
| running_xor | +0.015 (1.2σ, n.s.) | +0.126 (6.9σ) |
| running_sum_local | +0.009 (1.1σ, n.s.) | +0.110 (7.4σ) |
| sliding_window_max | +0.046 (4.9σ) | +0.112 (7.6σ) |
| copy_previous | −0.044 (4.6σ) | −0.107 (6.2σ), reset-rule confound |

(Original column updated 2026-11-07 with both 32B seed batches; the report's Table 6 still shows first-batch values,
which were −0.043/+0.011/+0.043. Values from `results_clean/_summary/split_heatmap_values.csv`.)

Position-local tasks at 32B gain **+0.17 to +0.22** (Figure 1 pooling, both batches: cyclic_shift +0.20,
random_permutation +0.22, conditional_shift +0.17).
So at 32B, only copy_previous is actually hurt. A reviewer will see this in Table 6 and Figure A7.

### The framing that holds at every scale

The robust quantity is **how much of the benefit a cut dependency cancels**, not the sign of the effect:

1. **Markers help position-local computation.** Gains at every model, but sized very differently (averaged over the three
   tasks, N ≤ 64): Qwen2.5-7B/14B/32B +0.09 / +0.17 / +0.20, Gemini +0.15, **Mistral +0.04, Phi +0.03** (small).
   Third-party task: ProcBench-Substitute +0.054 at 7B, +0.157 at 32B (5.9σ). A ceiling task (string copying) is a clean null.
2. **Cutting a dependency cancels that benefit.** At 32B the gain drops to roughly zero; at smaller models and on Gemini it
   turns into harm (sliding_window_max 7B −4.6σ, 14B −5.4σ; Gemini running_sum_local −0.133 at 5.6σ; CLRS-Text minimum
   −0.25 at 32B, −0.34 on Gemini).
3. **Causal test: severing the dependency restores the benefit.** Rewriting each task so its recurrence restarts at every
   marker brings the gain back (+0.11 to +0.13 at 32B) in 3 of 4 tasks; 6 of 7 task×model pairs move the predicted way.
   copy_previous is the exception, and its errors sit exactly on the rule switches the reset introduces.
4. **Mechanism: markers are attention sinks held by positional heads.** Marker attention rises with density at every length;
   the top marker-attending heads are positional (mean 95th percentile), not induction heads (31st). Blocking 10 of them
   from the markers removes up to 74% of the benefit at 7B and 32% at 32B. At a misaligned density, attention to the token the
   model actually needs drops from 0.54 to 0.10 (copy_previous, N=32), matching the accuracy dip.
5. **Practical guidance.** The best density falls as sequences grow (for cyclic_shift and copy_previous at 32B the densest
   setting wins through N=32, sparser settings after; sliding_window_max is noisier). Sparse markers
   (every 8th position) also run faster than no markers (0.44× Base wall-clock at N=128), and cost about a third of true
   divide-and-conquer while recovering much of its accuracy.

### Draft title and abstract

**The current title and abstract live in `paper_tmlr/main.tex`** (TMLR draft, 2026-11-07). The block below is the first
version, kept for reference; the paper's version corrects the Mistral/Phi wording and the "cancels" quantification.

**Title:** *Boundary Markers Help Until They Cut a Dependency: Structural Tokens as Attention Sinks in Sequence Transformation*

**Abstract (first version):**

> Language models lose accuracy quickly as deterministic sequence transformations grow longer. We study a cheap,
> single-call intervention: inserting a content-free marker token at a fixed interval and asking the model to copy it
> through. Across seven synthetic tasks organized by cross-position dependency, five open instruction-tuned models
> (Qwen2.5-7B/14B/32B, Mistral-7B, Phi-3.5-mini), a closed model (Gemini), and third-party tasks from ProcBench and
> CLRS-Text, the effect of a marker depends on what it lands on. On position-local tasks, markers raise token accuracy
> substantially (+0.16 to +0.22 at Qwen2.5-32B; up to +0.24 on Gemini). When each output depends on earlier inputs, that
> benefit is cancelled at 32B and becomes harm at smaller models and on Gemini. A causal test isolates the cause:
> rewriting each dependency task so its recurrence restarts at every marker restores the benefit in three of four tasks
> (+0.11 to +0.13 at 32B). Mechanistically, markers act as attention sinks held by positional rather than induction
> heads; blocking ten such heads from the markers removes up to 74% of the benefit at 7B and 32% at 32B, and at a
> misaligned density, attention to the token the model needs falls from 0.54 to 0.10. The best density decreases with
> sequence length, and sparse markers run faster than no markers at all. Markers are scaffolding for position-local
> computation and a hazard wherever they cut a dependency the task still needs.

*(Update the model list once the current-generation model run is in; see Section 3.)*

### Suggested figure order

The current main figures lead with density (Fig. 1). The headline is the helps/cancels split, so lead with that:

| New | Content | Currently |
|---|---|---|
| Fig. 1 | Helps vs. cancelled: Tier 1 vs. dependency tasks, across models + Gemini | Fig. 4 (compact it) |
| Fig. 2 | Causal reset: severing the dependency restores the benefit | Fig. 3 |
| Fig. 3 | Mechanism: attention sinks, head ablation, attention-to-target collapse | Fig. 2 |
| Fig. 4 | Practical: best density vs. length, cost vs. divide-and-conquer | Fig. 1 + A17 |

### Suggested section outline

1. Introduction
2. Setup: task tiers, the marker intervention, models, metrics (TAR/SAR), statistics
3. Markers help position-local tasks (Tier 1 across models, ProcBench, Gemini; string-copy null; vowel-tagging null)
4. Cutting a dependency cancels the benefit (Tier 2/3 across models and scale; CLRS-Text; Gemini)
5. Causal test: reset variants (and the copy_previous confound)
6. Mechanism: attention sinks, positional heads, head ablation, attention to the needed token
7. Practical guidance: density vs. length, cost, format tax, what about the marker matters (identity, regularity, reserved tokens fail)
8. Limitations
9. Conclusion

---

## 2. Claims to retire from the July draft

| July PDF says | Current evidence |
|---|---|
| "Delay the accuracy cliff"; SK error-accumulation model as the frame | Not part of the current report; demote to related work or appendix |
| Anchors are copy_previous and **running_max** | running_max replaced by sliding_window_max (its max saturates, so TAR rises with N spuriously) |
| "In all 43 task×model comparisons a parabola beats a line" | In-sample only. Held-out CV favors the parabola in 42–50% of Tier-1 curves and **0%** of Tier-2/3 curves (Table 1) |
| "Well-formedness tax in 9 of 9 tasks" | 6 of 7 at 32B; random_permutation is the exception (Table 7) |
| "Erases up to 62.6% of the benefit" (7B, N=64) | 7B: 2/35/33/63/74% at N=8–128; 32B: 0/0/11/32/28% (Fig. A14) |
| Composed substitution "capability wall" | Not in the current report; decide whether to keep it |
| "R² ≥ 0.97", "360 of 360 trials" | Re-verify against current data before reusing |
| Mechanism story built on 7B | Mechanism figures are now 32B, real letter tokenization |

---

## 3. Left to run

**Before submitting anywhere**

1. **Current-generation open models: QUEUED 2026-11-07.** Qwen3.8-27B (newest dense Qwen, Aug 2026) and
   Gemma-4-31B-it (Mar 2026), both ungated. `scripts/launch_new_models.sh`: per model a smoke job (download + validation)
   gating an 11-task array (3 position-local, 4 dependency, 4 reset), one task at a time per model.
   Jobs: smoke 40543558 → grid 40543559 (Qwen3.8); smoke 40543560 → grid 40543561 (Gemma 4).
   **Both smoke tests PASSED** (no thinking tags; completion 0.78–0.86). One trial per cell, so direction only:
   cyclic_shift Base→Bounds 0.51→0.80 (Qwen3.8), 0.55→0.79 (Gemma); copy_previous 0.91→0.69, 0.99→0.83.
   cyclic_shift (the only task that runs to N=256) was moved out of the 8 h arrays into 16 h jobs
   **40544929** (Qwen3.8) and **40544949** (Gemma): Qwen3.8 runs at the 72B run's speed (433 min), too close to 8 h.
   Array tasks 1–10 (N ≤ 64, ~2–3.5 h each) continue in 40543559 / 40543561.
   **Qwen3.8-27B is a hybrid**: 48 of 64 layers are Gated DeltaNet linear attention, 16 softmax. Its linear-attention
   kernels (`flash-linear-attention`) are not installed, so it runs on the slower PyTorch fallback (correct, just slow).
   The paper now says this (Setup + Limitations); it bears on the attention-sink mechanism claim.
2. **Second seed batch for 32B single-batch results: DONE 2026-11-07** (all four; jobs 40543549–40543552).
   Nothing changed qualitatively (all counts in Section 3 of the paper hold; z-scores rose). Figure 1 regenerated and the
   paper's numbers updated. Appendix per-model figures from the report still show first-batch curves. Jobs 40543549–40543552
   (`scripts/launch_32b_second_batch.sh`). Optional follow-up: the 32B reset variants are also single-batch (20/cell).
   After any of these finish: `python scripts/make_fig_split_heatmap.py`, `bash paper_tmlr/sync_figures.sh`, update numbers.
   All of these share the group's 4-A100 cap with the Promoter2Gene r3 jobs.
2b. **Token-budget fix: RUNNING (queued 2026-11-08).** The old budget int(1.5N)+20 is smaller than a correct
   k=1 answer (2N−1 tokens) once N > 42, so every k=1 output at N ≥ 64 was cut off (58 of 64 letters at N=64),
   for every local model and also ProcBench-Substitute and string copy. Gemini is unaffected (+50 margin).
   `runner.max_new_tokens` now gives k=1 a budget of 2N+20 (greedy decoding: other cells unchanged).
   `tasks/rerun_capped_k1.py` regenerates exactly those 5,886 rows (89 files) into `results_clean/_k1_rerun/`,
   checking each input rebuilds and each new answer starts with the old cut-off one (first rows: 100% match).
   Jobs 40550170–40550176 (`scripts/launch_k1_rerun.sh`, one per model). When done:
   `scripts/merge_k1_rerun.py --dry-run`, then without `--dry-run`, then regenerate Figure 1 and update the paper:
   significance counts (4 borderline cells), budget sentence in Setup, k=1 wall-clock claim (capped at N ≥ 64),
   density figure, and copy_previous's best density at N=32 (k=8 with both batches, not k=1).
3. ~~Vowel-tagging significance~~ **Done 2026-11-07** from per-trial data: 7B −0.083 (2.33σ, harm), 32B +0.012 (0.48σ, n.s.).
   The naturalistic check does **not** show the Tier-1 benefit; report it as a limitation, not support.

**Optional**

4. Gemini at 10 trials/cell (the 3-trial run cost ~$1.50, so ~$5).
5. One base (non-instruct) model on Tier 1 + copy_previous.
6. Overlong rate and boundary fidelity at full scale (currently a small sample only).

---

## 4. Limitations to state up front

- Open models are 2024-era and instruction-tuned only (until item 1 is run).
- The 32B reset variants are single-batch (20 trials/cell); Gemini is 3 trials/cell.
- The naturalistic vowel-tagging task shows no benefit (7B significantly hurt, 32B null): the Tier-1 benefit is shown for
  in-context rules, not yet for applying pretrained knowledge position by position.
- Dose-response: the quadratic shape passes held-out CV for only about half of Tier-1 curves and never for Tier-2/3.
  Present the density-vs-length shift descriptively, not as a fitted law.
- copy_previous's reset variant adds a rule-switching confound.
- Prompt sensitivity: the terse-cipher template reverses the effect at 32B (Table 5).
- A reserved special token as the marker fails at every length and both scales (Table 3).
- Mistral's Tier-1 benefit fades to about zero at long lengths.
- Attention analyses are tokenizer-dependent (validated against digits in Fig. A16).

---

## 5. Venue

- **Decision (2026-11-07): TMLR.** Draft in `paper_tmlr/` (official style; see its README). No fees; no length limit,
  though unusually long papers review more slowly; double-blind via OpenReview.
- **ICLR 2027:** missed (full-paper deadline was 2026-09-25).
- **Why TMLR** (journal, rolling submissions). Reviewers judge whether claims are supported by the evidence
  rather than novelty, which suits a careful, heavily caveated empirical paper; no strict page limit, so the appendix fits.
- **Conference alternative: COLM 2027** (deadline not yet announced; past years were late March). Best topical fit, and
  leaves time for the new-model runs. ACL's January ARR cycle and ICML 2027 (both ~January, unconfirmed) are likely too soon.
- Post an arXiv preprint once the 32B second batch and the abstract are settled.
