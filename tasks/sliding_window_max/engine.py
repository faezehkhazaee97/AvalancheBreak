"""Tier 3 candidate replacement for running_max: bounded k-step-window
maximum instead of a cumulative (unbounded) maximum. output[i] = the
largest digit among the last WINDOW_SIZE positions (this one included,
truncated near the start where fewer than WINDOW_SIZE digits exist yet).

Why not running_max: its cumulative max is monotonic and absorbing --
once the alphabet's ceiling digit appears (fast: expected within ~
alphabet_size positions), every later position trivially repeats it.
Verified empirically: 98.2% of output positions are already at the
ceiling by N=64 (alphabet=2), and TAR on the genuinely informative
(non-ceiling) positions actually DECLINES with N even though pooled TAR
rises -- a task-design artifact, not a real accuracy-cliff signal.

Why not running_majority (the first replacement candidate tried): it
requires an exact, persistent (count0, count1) accumulator -- a single
miscount anywhere corrupts every later position, the same cascading-
arithmetic-error failure mode that makes running_sum_local/modsum hard.
Screened at Qwen2.5-7B and failed (inconsistent, one length below chance).

Why a fixed WINDOW_SIZE fixes both problems at once: the output only
depends on the last WINDOW_SIZE raw input digits, not a growing summary,
so (a) it can't saturate -- old high values fall out of the window as it
slides, keeping the statistic genuinely live at every length; (b) an
error only contaminates output for up to WINDOW_SIZE positions, not
forever, since the very next slide can recover from it; (c) difficulty is
constant across N (position 5 and position 60 both only look back
WINDOW_SIZE steps) instead of escalating; (d) unlike majority there is no
tie to break -- "biggest digit value among these WINDOW_SIZE positions"
has a single unambiguous answer whenever at least two candidate digits
differ, and repeated max-valued digits don't need a tiebreak since the
VALUE (not the identity/position) is what's reported.

Base competence for this specific "compare k small digits, report the
largest" operation is not a new gamble: running_max's own saturated-
position accuracy (0.67-0.97 at Qwen2.5-7B) already shows the model
handles exactly this operation well when the task calls for it -- the
problem there was never digit-comparison competence, only the pooled
metric's saturation artifact.

Not yet grid-verified -- this is exactly what the Base-only screen is for.

WINDOW_SIZE=3, reverted from a brief WINDOW_SIZE=4 experiment: window=4
would have let k=4 (a standard, included boundary density -- k=3 itself is
excluded from this study's sweep) serve as a clean "boundaries land
exactly at window edges" condition, matching local_block_reversal's
aligned/misaligned logic. But screened head-to-head, window=4 was
measurably HARDER for Qwen2.5-7B than window=3 -- weaker z-scores at
every alphabet size, alphabet=4 dropping below the naive baseline by
N=16 instead of N=32 -- and category analysis showed why: at window=4 the
model leans harder on lazy constant-output guessing (which happens to
score well against the skewed true-value distribution) instead of
genuine tracking. That competence cost outweighs the alignment-condition
convenience, which was a nice-to-have, not a requirement -- reverted to
window=3.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "sliding_window_max"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [4, 8]  # alphabet=2 dropped: screened weak, model exploits the skewed
# output distribution via lazy constant-guessing (65% of alphabet=2 trials) instead
# of genuine tracking, matching composed_permutation's own precedent for excluding
# alphabet=2 from its sweep
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [4, 8, 16, 32, 64]

WINDOW_SIZE = 3


def _compute_sliding_max(digit_str, window_size):
    n = len(digit_str)
    output = []
    for i in range(n):
        start = max(0, i - window_size + 1)
        window = digit_str[start : i + 1]
        output.append(max(window))  # char comparison == numeric order for single ASCII digits
    return "".join(output)


def build_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    input_str_base = "".join(str(rng.randint(0, alphabet_size - 1)) for _ in range(N))
    output_str_base = _compute_sliding_max(input_str_base, WINDOW_SIZE)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Sliding-window maximum.
Alphabet: 0-{alphabet_size - 1}
Rule: At each position, output the LARGEST digit among this position and the {WINDOW_SIZE - 1} position(s) immediately before it (a window of the last {WINDOW_SIZE} digits, including the current one).
Near the very start of the sequence, where fewer than {WINDOW_SIZE} digits exist yet, use only the digits that exist so far.
Example (window={WINDOW_SIZE}): Input 3021 -> Output 3332 (position 0: window=[3], max=3. position 1: window=[3,0], max=3. position 2: window=[3,0,2], max=3. position 3: window=[0,2,1] (oldest '3' has fallen out of the window), max=2.)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT include them in the window or count them as digits. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
