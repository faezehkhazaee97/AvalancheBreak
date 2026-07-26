"""Tier 2 EASY ANCHOR (current + immediately-previous): pure copy, no
computation at all. output[0] = input[0] (no previous); output[i] =
input[i-1] for i >= 1 -- the whole output is the input shifted right by
one position, first character unchanged.

Why this task exists: running_xor's 7B baseline sits near or below random
chance (0.31-0.49) in most cells -- meaning our flagship Tier-2 task risks
the same "task too hard, nothing to scaffold" confound composed_permutation
revealed in Tier 1. This task isolates the OTHER half of Tier 2's claim:
can boundaries help pure one-step attention/copying, with zero arithmetic
difficulty in the way? Baseline should be high (copying one position back
is about as easy as a Tier-2 task gets), giving a clean "genuine
competence" anchor to compare against running_xor/running_sum_local's
harder, sometimes near-chance baselines.

Binary alphabet, same convention as running_xor, for direct comparability.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "copy_previous"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True


def _compute_shift(binary_str):
    output = [binary_str[0]]
    for i in range(1, len(binary_str)):
        output.append(binary_str[i - 1])
    return "".join(output)


def build_prompt(N, k, rng, axis_val=None):
    input_str_base = "".join(str(rng.choice([0, 1])) for _ in range(N))
    output_str_base = _compute_shift(input_str_base)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Copy the previous input digit.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the PREVIOUS input digit (not the current one).
The first digit is copied as-is (no previous input to copy).
Each subsequent digit: output[i] = input[i-1]
Example: Input 1011 -> Output 1101 (calculation: 1 (copied), then 1, 0, 1 -- each shifted from the position before)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
