"""Tier 3 (global state): running sum modulo M, direction-aligned (cumulative
sum flows left-to-right, matching generation direction). Modulus sweep
{2,4,8} -- the axis actually cited by the paper's appendix figure.

Task logic ported unchanged from run_modsum_modulus_sweep.py. Scoring goes
through tasks.common.scoring instead of that script's
re.match(r'^(\\d+)', pred) -- the same Finding A truncation-at-'|' fix as
running_sum_local. The paper's in-text modsum table (a separate script,
run_modsum_templated_v2.py) already used the correct strip-'|' convention
this module now standardizes on; only the appendix-figure source was
contaminated.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "modsum"
AXIS_NAME = "Modulus"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True


def _compute_cumulative_sum_mod(digit_str, modulus):
    output = []
    cumsum = 0
    for ch in digit_str:
        cumsum = (cumsum + int(ch)) % modulus
        output.append(str(cumsum))
    return "".join(output)


def build_prompt(N, k, rng, axis_val):
    modulus = axis_val
    input_str_base = "".join(str(rng.randint(0, modulus - 1)) for _ in range(N))
    output_str_base = _compute_cumulative_sum_mod(input_str_base, modulus)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running sum modulo {modulus}.
Alphabet: 0-{modulus - 1}
Modulus: {modulus}
Rule: At each position, output the cumulative sum of all digits so far, modulo {modulus}.
Sum all digits from the start up to this position, divide by {modulus}, output the remainder.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
