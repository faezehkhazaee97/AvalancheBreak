"""sliding_window_max, but boundaries are real reset points instead of a
pure visual overlay. In the original sliding_window_max, output_str_base
is computed on the FULL continuous sequence before boundaries are spliced
into both strings -- so the window can still slide across a "|" (the model
just has to skip the bar character while collecting its 3 most recent
digits), meaning the true dependency still crosses the boundary even
though one sits there. Here, the window is redefined to never reach
before the most recent boundary at all -- exactly like the very start of
the sequence, where fewer than window_size digits exist yet and the rule
already says "use only the digits that exist so far."

This is the same causal-necessity test already run for running_xor,
running_sum_local, copy_previous, and reverse_xor: if dense-boundary harm
comes from breaking a dependency the model still needs, removing that
need here should shrink or reverse the harm. If harm persists anyway,
boundary insertion is disruptive on its own, independent of whether
anything actually needs to cross it.

At k=0 (Base) this reduces to sliding_window_max exactly, since every
position's "most recent boundary" is the start of the sequence -- same as
the original task's own start-of-sequence handling.

Boundary-start position math mirrors tasks.common.boundary.insert_boundaries
exactly: a "|" is inserted between index i and i+1 iff (i+1) % k == 0 and
i != len-1, so position j's "most recent boundary" is at chunk_start =
(j // k) * k when k > 0.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "sliding_window_max_reset"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [4, 8]  # same exclusion as sliding_window_max: alphabet=2 screened weak
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [4, 8, 16, 32, 64]

WINDOW_SIZE = 3


def _compute_sliding_max_reset(digit_str, window_size, k):
    n = len(digit_str)
    output = []
    for i in range(n):
        chunk_start = (i // k) * k if k > 0 else 0
        start = max(chunk_start, i - window_size + 1)
        window = digit_str[start : i + 1]
        output.append(max(window))
    return "".join(output)


def build_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    input_str_base = "".join(str(rng.randint(0, alphabet_size - 1)) for _ in range(N))
    output_str_base = _compute_sliding_max_reset(input_str_base, WINDOW_SIZE, k)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Sliding-window maximum, reset at boundaries.
Alphabet: 0-{alphabet_size - 1}
Rule: At each position, output the LARGEST digit among this position and the {WINDOW_SIZE - 1} position(s) immediately before it (a window of the last {WINDOW_SIZE} digits, including the current one).
Near the very start of the sequence, where fewer than {WINDOW_SIZE} digits exist yet, use only the digits that exist so far.
IMPORTANT: The input contains vertical bars (|), structural boundaries. Immediately after each boundary, treat that position exactly like the very start of the input -- the window resets and only accumulates digits seen since that boundary, never reaching back across it, until {WINDOW_SIZE} digits have accumulated in the current segment or the next boundary arrives.
Example (window={WINDOW_SIZE}): Input 3021|1247 -> Output 3332|1247 (first segment: same as the no-boundary rule. second segment restarts fresh: position 0 of the segment has window=[1], max=1; position 1 has window=[1,2], max=2; position 2 has window=[1,2,4], max=4; position 3 has window=[2,4,7] (window is now full within this segment, but never reaches back into the first segment), max=7.)
Constraint: Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """
    return input_bounded, output_bounded, prompt
