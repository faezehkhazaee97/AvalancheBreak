"""Tier 2 (current + immediately-NEXT), direction-reversed: same binary XOR
chain as running_xor, but looking FORWARD instead of backward.
output[N-1] = input[N-1] (no next); output[i] = input[i] XOR input[i+1].

Why this task exists: running_xor's dependency (output[i] needs input[i-1],
already generated) flows the SAME direction as autoregressive generation.
This task's dependency (output[i] needs input[i+1], not yet "generated" but
already visible in the prompt) flows the OPPOSITE direction. This is the
same distinction floated earlier to explain why addition (carries flow
right-to-left, opposite generation) and modsum (cumulative sum flows
left-to-right, same as generation) behaved so differently in Tier 3 --
never directly tested there. Same rule family as running_xor (XOR, binary
alphabet, identical boundary/scoring mechanics), so this isolates direction
alone as the variable, holding everything else fixed.

Logic mirrors tasks/running_xor/engine.py exactly, with the dependency
direction reversed.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "reverse_xor"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True


def _compute_xor_chain_reverse(binary_str):
    n = len(binary_str)
    output = [str(int(binary_str[i]) ^ int(binary_str[i + 1])) for i in range(n - 1)]
    output.append(binary_str[-1])
    return "".join(output)


def build_prompt(N, k, rng, axis_val=None):
    input_str_base = "".join(str(rng.choice([0, 1])) for _ in range(N))
    output_str_base = _compute_xor_chain_reverse(input_str_base)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running XOR with next input.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the NEXT input digit.
The last digit is copied as-is (no next input to XOR with).
Each other digit: output[i] = input[i] XOR input[i+1]
Example: Input 1011 → Output 1101 (calculation: 1⊕0=1, 0⊕1=1, 1⊕1=0, 1)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
