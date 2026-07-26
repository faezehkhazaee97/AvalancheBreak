"""Tier 2 (current + immediately-previous): running sum, local window, with
a Modulus sweep {2,4,8}. output[0] = input[0] (no previous);
output[i] = (input[i] + input[i-1]) mod M.

Task logic ported unchanged from run_running_sum_local_modulus_sweep.py.
Scoring goes through tasks.common.scoring instead of that script's
re.match(r'^(\\d+)', pred) -- THIS IS THE FIX for Finding A: the old regex
truncated the recorded output at the first '|' while the ground truth still
had bars in it, so every position after the first boundary auto-scored
wrong. This task previously had no clean data anywhere in the repo.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "running_sum_local"
AXIS_NAME = "Modulus"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True


def _compute_running_sum_local(digit_str, modulus):
    output = [digit_str[0]]
    for i in range(1, len(digit_str)):
        output.append(str((int(digit_str[i]) + int(digit_str[i - 1])) % modulus))
    return "".join(output)


def build_prompt(N, k, rng, axis_val):
    modulus = axis_val
    input_str_base = "".join(str(rng.randint(0, modulus - 1)) for _ in range(N))
    output_str_base = _compute_running_sum_local(input_str_base, modulus)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running sum with previous input (local window).
Alphabet: 0-{modulus - 1} (digits)
Modulus: {modulus}
Rule: At each position, output the sum of the current input digit and the previous input digit, modulo {modulus}.
The first digit is copied as-is (no previous input to add).
Each subsequent digit: output[i] = (input[i] + input[i-1]) mod {modulus}
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
