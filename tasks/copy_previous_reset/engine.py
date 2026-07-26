"""copy_previous, but boundaries are real reset points instead of a pure
visual overlay. In the original copy_previous, output[i] = input[i-1]
regardless of where boundaries fall -- the true dependency still crosses
the "|" even when one sits there. Here, the position immediately after
each boundary has nothing valid to copy from across the boundary, so it
falls back to copying itself (output[i] = input[i]), exactly like
position 0 does in the original task.

See tasks/running_xor_reset/engine.py for the full rationale: this isolates
whether dense-boundary harm comes from breaking a still-needed dependency,
versus boundary insertion being disruptive on its own.

At k=0 (Base) this reduces to copy_previous exactly.

Boundary-start position math: position j is "immediately after a boundary"
iff j % k == 0 and j > 0, matching tasks.common.boundary.insert_boundaries.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "copy_previous_reset"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True


def _compute_shift_reset(binary_str, k):
    n = len(binary_str)
    output = [binary_str[0]]
    for i in range(1, n):
        is_chunk_start = k > 0 and i % k == 0
        if is_chunk_start:
            output.append(binary_str[i])
        else:
            output.append(binary_str[i - 1])
    return "".join(output)


def build_prompt(N, k, rng, axis_val=None):
    input_str_base = "".join(str(rng.choice([0, 1])) for _ in range(N))
    output_str_base = _compute_shift_reset(input_str_base, k)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Copy the previous input digit, reset at boundaries.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the PREVIOUS input digit (not the current one).
The first digit is copied as-is (no previous input to copy).
Each subsequent digit: output[i] = input[i-1]
IMPORTANT: The input contains vertical bars (|), structural boundaries. Immediately after each boundary, treat that position exactly like the first digit of the whole input -- copy IT as-is (nothing before a boundary counts as "previous"), then resume the shift-by-one rule from there until the next boundary.
Example: Input 1011|0010 → Output 1101|0001 (first block: 1 copied, then 1,0,1 shifted from the position before; second block restarts: 0 copied, then 0,0,1 shifted from the position before, within that block only)
Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
