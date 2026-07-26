"""Tier 2 (current + immediately-previous): running XOR chain, binary alphabet.
output[0] = input[0] (no previous); output[i] = input[i] XOR input[i-1].

Logic ported unchanged from the proven run_running_xor_structured_boundaries.py
(the only engine this task has -- no sweep/templated variant exists); boundary
insertion now goes through tasks.common.boundary (fixes k>=N degenerate cells).
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "running_xor"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True


def _compute_xor_chain(binary_str):
    output = [binary_str[0]]
    for i in range(1, len(binary_str)):
        output.append(str(int(binary_str[i]) ^ int(binary_str[i - 1])))
    return "".join(output)


def build_prompt(N, k, rng, axis_val=None):
    input_str_base = "".join(str(rng.choice([0, 1])) for _ in range(N))
    output_str_base = _compute_xor_chain(input_str_base)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running XOR with previous input.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the previous input digit.
The first digit is copied as-is (no previous input to XOR with).
Each subsequent digit: output[i] = input[i] XOR input[i-1]
Example: Input 1011 → Output 1110 (calculation: 1, 0⊕1=1, 1⊕0=1, 1⊕1=0)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
