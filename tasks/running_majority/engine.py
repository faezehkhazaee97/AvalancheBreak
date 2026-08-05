"""Tier 3 EASY ANCHOR (full-prefix dependency), replacement candidate for
running_max. output[i] = whichever binary digit (0 or 1) is currently more
frequent in input[0..i]; ties go to the digit that just arrived. Binary
alphabet only (no Alphabet_Size sweep), matching copy_previous's own
precedent as the Tier-2 easy anchor -- keeps the per-step computation to
one running score and one comparison, the same shape as running_max's own
"track one value, compare-and-replace" design intent, rather than the
multi-category tally a larger-alphabet majority would need.

Why running_max needed replacing: its output is a MONOTONIC, ABSORBING
statistic -- once the alphabet's max digit appears (fast: expected within
~alphabet_size positions), every subsequent position trivially repeats it.
Verified empirically: by N=64 at alphabet=2, 98.2% of all output positions
are already at the ceiling value, and TAR on the genuinely informative
(non-saturated) positions actually DECLINES with N (0.19->0.05 at
alphabet=8) even though pooled/aggregate TAR rises with N -- the opposite
of the accuracy-cliff pattern central to this study, entirely a task-design
artifact.

running_majority does not have this problem: for a binary sequence of
i.i.d. draws, "is count(1) > count(0)?" is a simple random walk, which is
recurrent -- it crosses zero (and therefore flips the current leader)
infinitely often almost surely, so the statistic never permanently settles
the way a running max does. Verified empirically (P(output constant for
the entire last quarter of the sequence), simulated over 500 trials):
running_max is 93.4-100% frozen by N=8 and gets WORSE (more frozen) as N
grows (99.8% by N=64); running_majority stays in the 41-72% range at every
tested length and, if anything, gets LESS frozen at larger alphabets --
the two statistics behave oppositely.

Not yet grid-verified for baseline model competence -- this is exactly
what the Base-only screen (tasks/common/resumable_runner.run_screen) is
for, matching the same screening convention already used for the
external-tasks suite (string_copy, procbench_substitute, flip_flop).
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "running_majority"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [4, 8, 16, 32, 64]


def _compute_running_majority(binary_str):
    count0 = 0
    count1 = 0
    output = []
    for ch in binary_str:
        if ch == "0":
            count0 += 1
        else:
            count1 += 1
        if count1 > count0:
            leader = "1"
        elif count0 > count1:
            leader = "0"
        else:
            leader = ch  # tie -> the digit that just arrived
        output.append(leader)
    return "".join(output)


def build_prompt(N, k, rng, axis_val=None):
    input_str_base = "".join(str(rng.choice([0, 1])) for _ in range(N))
    output_str_base = _compute_running_majority(input_str_base)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running majority vote.
Alphabet: 0, 1 (binary digits)
Rule: At each position, look at every digit from the start of the sequence up to and including this position. Output whichever digit (0 or 1) has appeared MORE TIMES so far.
Tie-breaking: if 0 and 1 have appeared the SAME number of times so far, output the digit at the CURRENT position (the one you just read).
Example: Input 0011 -> Output 0001 (after "0": 0 leads, output 0. After "00": 0 leads, output 0. After "001": tied 1-1, current digit is 1, output 1. After "0011": 1 now leads (2 vs 1), output 1.)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT count them as digits. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
