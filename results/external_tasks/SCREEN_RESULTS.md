# External-tasks Base-only screen (Qwen2.5-7B-Instruct, 10 trials/cell)

Rule applied: drop a task if Base TAR is not significantly above chance at *any* tested length.

## string_copy — SURVIVES

Chance = 1/alphabet_size. Above chance at every (length, alphabet) cell tested, most at TAR≈0.9–1.0
(z-scores in the hundreds at larger alphabets, since correct copying should be near-trivial for a
competent model). No surprises.

## procbench_substitute — SURVIVES

Chance ≈ 1/35 (36-symbol alphabet minus the source character). Above chance at every length
(N=4: TAR=0.350, z=2.70, p=0.0035 through N=64: TAR=0.053, z=1.99, p=0.024) — real, if modest,
competence throughout, and it degrades with length in the expected direction.

## flip_flop_high — DROP (with a caveat, see below)

## flip_flop_low — DROP (with a caveat, see below)

Both flip-flop variants come back **below** chance (0.5) at every length, not just at it
(flip_flop_high: 0.20–0.43 TAR across N=4–64; flip_flop_low: 0.13–0.33). Checked raw outputs
before accepting this — it is not a scoring bug. The model is frequently emitting a single
character (e.g. `"i"` or `"0"`) instead of the full N-character register-state stream, then
stopping generation entirely:

```
Input='i0rr'  Expected='0000'  Predicted='i'   TAR=0.0
Input='rir1'  Expected='0001'  Predicted='0'   TAR=0.25
Input='irr0'  Expected='0000'  Predicted='0'   TAR=0.25
```

`max_new_tokens` was not the constraint (26 tokens were available for N=4; the model stopped on
its own). This looks like a completion/prompt-clarity failure — the model may be treating the
task like the original paper's "answer the read query" framing rather than "emit one state
character per input position" — rather than a genuine incapacity to track a single bit.

**Prompt fix attempted (2026-07-27), re-screened, still drops.** Added one explicit sentence to
`tasks/common/flip_flop_shared.py`'s prompt ("Output length: your output must contain one
character for EVERY position in the input, including every 'r' and 'i' -- do not stop early at
an 'r'; it is not a final query, just another position to emit a value for. The output is
exactly as long as the input.") — a single, narrow, targeted change, nothing else about the task
definition/alphabet/density touched. Re-ran the 10-trial Base-only screen at Qwen2.5-7B-Instruct
only, per the agreed plan (fix once, re-screen, stop regardless of outcome — no escalation to
32B, no further wording iteration).

Result: still below chance at every length, for both variants:

| Task | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|
| flip_flop_high (z vs 0.5) | 0.400 (z=−0.63) | 0.338 (z=−1.03) | 0.375 (z=−0.79) | 0.338 (z=−1.03) | 0.439 (z=−0.39) |
| flip_flop_low (z vs 0.5) | 0.125 (z=−2.37) | 0.138 (z=−2.29) | 0.125 (z=−2.37) | 0.356 (z=−0.91) | 0.139 (z=−2.28) |

Raw outputs after the fix show the same completion failure persists (model still truncates to a
few characters, sometimes still echoing 'i'/'r' literally):
```
Input='iri01i0r'  Expected='00001100'  Predicted='011111'   TAR=0.375
Input='iiii01r1'  Expected='00000111'  Predicted='i11111'   TAR=0.125
```
The hypothesized cause (query-framing confusion from the original paper's task design) either
wasn't the true cause, or the fix wasn't strong enough to override it. Per the agreed stopping
condition, this is now final: **both variants stay dropped.** No further prompt iteration or
32B escalation.

## Verdict

Full 20-trial grid runs on: **string_copy**, **procbench_substitute**.
flip_flop_high / flip_flop_low: dropped after both the original screen and one targeted
prompt-fix re-screen. Report as an attempted-and-unsuccessful external Tier-2 check, not a
silent omission.
