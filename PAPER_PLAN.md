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
| running_sum_local | +0.011 (1.0σ, n.s.) | +0.110 (7.4σ) |
| sliding_window_max | +0.043 (3.3σ) | +0.112 (7.6σ) |
| copy_previous | −0.043 (3.1σ) | −0.107 (6.2σ), reset-rule confound |

Position-local tasks at 32B gain **+0.16 to +0.22** (cyclic_shift +0.19, random_permutation +0.22, conditional_shift +0.16; Table 7).
So at 32B, only copy_previous is actually hurt. A reviewer will see this in Table 6 and Figure A7.

### The framing that holds at every scale

The robust quantity is **how much of the benefit a cut dependency cancels**, not the sign of the effect:

1. **Markers help position-local computation.** Large, significant gains at every model (Fig. 4a, A1–A3), on Gemini
   (cyclic_shift +0.19 at 5.0σ, random_permutation +0.24 at 5.5σ), and on a third-party task (ProcBench-Substitute: +0.054 at 7B,
   +0.157 at 32B, 5.9σ). A ceiling task (string copying) is a clean null.
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

**Title:** *Boundary Markers Help Until They Cut a Dependency: Structural Tokens as Attention Sinks in Sequence Transformation*

**Abstract (draft, ~220 words):**

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

1. **A current-generation open model.** All open models are from 2024. Run the core grid plus reset variants
   (cyclic_shift, random_permutation, copy_previous, sliding_window_max, running_sum_local, + `_reset`) on a current Qwen
   release and one other family. *Needs a decision: which models.*
2. **Second seed batch for 32B single-batch results.** Ready to go: `bash scripts/launch_32b_second_batch.sh`
   (4 A100 jobs, new trial indices, can't overwrite batch 1). Held back because the Promoter2Gene r3 jobs were using the
   same `quonbiogrp` A100 allocation on 2026-11-07. After it finishes, regenerate the figures that glob
   `Qwen2.5-32B-Instruct*.csv` and update the report numbers.
3. ~~Vowel-tagging significance~~ **Done 2026-11-07** from per-trial data: 7B −0.083 (2.33σ, harm), 32B +0.012 (0.48σ, n.s.).
   The naturalistic check does **not** show the Tier-1 benefit; report it as a limitation, not support.

**Optional**

4. Gemini at 10 trials/cell (the 3-trial run cost ~$1.50, so ~$5).
5. One base (non-instruct) model on Tier 1 + copy_previous.
6. Overlong rate and boundary fidelity at full scale (currently a small sample only).

---

## 4. Limitations to state up front

- Open models are 2024-era and instruction-tuned only (until item 1 is run).
- Several 32B cells are single-batch; sliding_window_max is 20 trials/point; Gemini is 3 trials/cell.
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

- **ICLR 2027:** missed (full-paper deadline was 2026-09-25).
- **Recommended: TMLR** (journal, rolling submissions). Reviewers judge whether claims are supported by the evidence
  rather than novelty, which suits a careful, heavily caveated empirical paper; no strict page limit, so the appendix fits.
- **Conference alternative: COLM 2027** (deadline not yet announced; past years were late March). Best topical fit, and
  leaves time for the new-model runs. ACL's January ARR cycle and ICML 2027 (both ~January, unconfirmed) are likely too soon.
- Post an arXiv preprint once the 32B second batch and the abstract are settled.
