"""Naturalistic prefix-dependent task (reviewer question): a running checksum.
output[i] = (sum of input[0..i]) mod 10 over decimal digits. Every position depends on the whole
prefix, so a marker boundary always separates needed state."""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "running_checksum"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True


def _checksum(s):
    out, total = [], 0
    for ch in s:
        total = (total + int(ch)) % 10
        out.append(str(total))
    return "".join(out)


def build_prompt(N, k, rng, axis_val=None):
    input_str_base = "".join(str(rng.randint(0, 9)) for _ in range(N))
    output_str_base = _checksum(input_str_base)
    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)
    prompt = f"""You are a deterministic sequence transformer.
Task: Running checksum.
Alphabet: 0-9 (digits)
Rule: At each position, output the sum of all input digits up to and including this position, modulo 10.
Each digit: output[i] = (input[0] + input[1] + ... + input[i]) mod 10
Example: Input 4739 -> Output 4143 (4, 4+7=11->1, 11+3=14->4, 14+9=23->3)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """
    return input_bounded, output_bounded, prompt
