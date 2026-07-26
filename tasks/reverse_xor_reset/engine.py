"""reverse_xor, but boundaries are real reset points instead of a pure
visual overlay. reverse_xor's dependency looks FORWARD (output[i] needs
input[i+1]), the opposite direction from running_xor. So the position that
needs resetting here is the one immediately BEFORE each boundary -- it has
nothing valid to XOR with across the boundary, so it falls back to copying
itself (output[i] = input[i]), exactly like the last position of the whole
sequence does in the original task.

See tasks/running_xor_reset/engine.py for the full rationale.

At k=0 (Base) this reduces to reverse_xor exactly.

Boundary-end position math mirrors tasks.common.boundary.insert_boundaries
exactly: a "|" is inserted between index i and i+1 iff (i+1) % k == 0 and
i != len-1, so position i is "immediately before a boundary" under that
same condition.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "reverse_xor_reset"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True


def _compute_xor_chain_reverse_reset(binary_str, k):
    n = len(binary_str)
    output = []
    for i in range(n):
        is_chunk_end = k > 0 and (i + 1) % k == 0 and i != n - 1
        if i == n - 1 or is_chunk_end:
            output.append(binary_str[i])
        else:
            output.append(str(int(binary_str[i]) ^ int(binary_str[i + 1])))
    return "".join(output)


def build_prompt(N, k, rng, axis_val=None):
    input_str_base = "".join(str(rng.choice([0, 1])) for _ in range(N))
    output_str_base = _compute_xor_chain_reverse_reset(input_str_base, k)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running XOR with next input, reset at boundaries.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the NEXT input digit.
The last digit is copied as-is (no next input to XOR with).
Each other digit: output[i] = input[i] XOR input[i+1]
IMPORTANT: The input contains vertical bars (|), structural boundaries. Immediately BEFORE each boundary, treat that position exactly like the last digit of the whole input -- copy it as-is (nothing after a boundary counts as "next" for it), then the rule resumes normally on the other side of the boundary.
Example: Input 1011|0010 → Output 1101|0110 (first block: 1⊕0=1, 0⊕1=1, 1⊕1=0, 1 copied since it's right before the boundary; second block: 0⊕0=0, 0⊕1=1, 1⊕0=1, 0 copied since it's the last digit)
Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
