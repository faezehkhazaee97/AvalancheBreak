"""Tier 3 EASY ANCHOR (global state): running maximum, direction-aligned
(scan flows left-to-right, matching generation direction, same as modsum).
Modulus-style Alphabet_Size sweep {2,4,8}, mirroring modsum's own axis for
direct comparability. output[i] = max(input[0..i]).

Why this task exists: modsum's 7B baseline is low (0.17-0.29 across
lengths) -- close enough to chance that its weak/mixed boundary effect
could be the same "task too hard, nothing to scaffold" confound
composed_permutation revealed in Tier 1, rather than a genuine property of
global-state dependency. This task still needs the WHOLE prefix like
modsum does (output[i] depends on input[0..i], not just input[i-1]) but is
a much easier computation -- track one running value, compare-and-replace
-- so baseline competence should be high. A clean "genuine competence"
Tier-3 anchor to compare against modsum's harder, closer-to-chance
baseline.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "running_max"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True


def _compute_running_max(digit_str):
    output = []
    running_max = -1
    for ch in digit_str:
        running_max = max(running_max, int(ch))
        output.append(str(running_max))
    return "".join(output)


def build_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    input_str_base = "".join(str(rng.randint(0, alphabet_size - 1)) for _ in range(N))
    output_str_base = _compute_running_max(input_str_base)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running maximum.
Alphabet: 0-{alphabet_size - 1}
Rule: At each position, output the LARGEST digit seen so far, including the current position.
Scan left to right; once a digit is output, it never decreases.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
