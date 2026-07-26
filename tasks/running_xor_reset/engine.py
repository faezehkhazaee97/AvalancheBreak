"""running_xor, but boundaries are real reset points instead of a pure
visual overlay. In the original running_xor, output_str_base is computed
on the FULL continuous sequence before boundaries are spliced into both
strings -- so the true dependency (output[i] needs input[i-1]) still
crosses the "|" even though a boundary sits there, meaning the model still
has to reach across it. Here, the position immediately after each boundary
is redefined to need nothing from before the boundary at all -- exactly
like position 0 (no previous input, so output[i] = input[i]).

This isolates a confound the original result can't rule out: does dense
boundary insertion hurt because it breaks a dependency the model still
needs, or is inserting "|" characters disruptive on its own, independent
of whether anything actually needs to cross them? If harm disappears (or
reverses) here, that confirms "breaks a needed dependency" as the real
mechanism. If harm persists anyway, something else is going on.

At k=0 (Base) this reduces to running_xor exactly, since no position is a
"boundary start" -- computation matches the original XOR chain everywhere.

Boundary-start position math mirrors tasks.common.boundary.insert_boundaries
exactly: a "|" is inserted between index i and i+1 iff (i+1) % k == 0 and
i != len-1, so position j is "immediately after a boundary" iff j % k == 0
and j > 0.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "running_xor_reset"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True


def _compute_xor_chain_reset(binary_str, k):
    n = len(binary_str)
    output = []
    for i in range(n):
        is_chunk_start = i > 0 and k > 0 and i % k == 0
        if i == 0 or is_chunk_start:
            output.append(binary_str[i])
        else:
            output.append(str(int(binary_str[i]) ^ int(binary_str[i - 1])))
    return "".join(output)


def build_prompt(N, k, rng, axis_val=None):
    input_str_base = "".join(str(rng.choice([0, 1])) for _ in range(N))
    output_str_base = _compute_xor_chain_reset(input_str_base, k)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running XOR with previous input, reset at boundaries.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the previous input digit.
The first digit is copied as-is (no previous input to XOR with).
Each subsequent digit: output[i] = input[i] XOR input[i-1]
IMPORTANT: The input contains vertical bars (|), structural boundaries. Immediately after each boundary, treat that position exactly like the first digit of the whole input -- copy it as-is (nothing before a boundary counts as "previous"), then resume the XOR rule from there until the next boundary.
Example: Input 1011|0010 → Output 1110|0011 (first block: 1 copied, 0⊕1=1, 1⊕0=1, 1⊕1=0; second block restarts: 0 copied, 0⊕0=0, 1⊕0=1, 0⊕1=1)
Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
