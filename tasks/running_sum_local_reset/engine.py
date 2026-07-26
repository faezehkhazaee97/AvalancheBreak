"""running_sum_local, but boundaries are real reset points instead of a
pure visual overlay. In the original task, output[i] = (input[i] +
input[i-1]) mod M regardless of where boundaries fall. Here, the position
immediately after each boundary has nothing valid to add from across the
boundary, so it falls back to copying itself (output[i] = input[i]),
exactly like position 0 does in the original task.

See tasks/running_xor_reset/engine.py for the full rationale.

At k=0 (Base) this reduces to running_sum_local exactly.

Boundary-start position math: position j is "immediately after a boundary"
iff j % k == 0 and j > 0, matching tasks.common.boundary.insert_boundaries.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "running_sum_local_reset"
AXIS_NAME = "Modulus"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True


def _compute_running_sum_local_reset(digit_str, modulus, k):
    n = len(digit_str)
    output = [digit_str[0]]
    for i in range(1, n):
        is_chunk_start = k > 0 and i % k == 0
        if is_chunk_start:
            output.append(digit_str[i])
        else:
            output.append(str((int(digit_str[i]) + int(digit_str[i - 1])) % modulus))
    return "".join(output)


def build_prompt(N, k, rng, axis_val):
    modulus = axis_val
    input_str_base = "".join(str(rng.randint(0, modulus - 1)) for _ in range(N))
    output_str_base = _compute_running_sum_local_reset(input_str_base, modulus, k)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running sum with previous input (local window), reset at boundaries.
Alphabet: 0-{modulus - 1} (digits)
Modulus: {modulus}
Rule: At each position, output the sum of the current input digit and the previous input digit, modulo {modulus}.
The first digit is copied as-is (no previous input to add).
Each subsequent digit: output[i] = (input[i] + input[i-1]) mod {modulus}
IMPORTANT: The input contains vertical bars (|), structural boundaries. Immediately after each boundary, treat that position exactly like the first digit of the whole input -- copy it as-is (nothing before a boundary counts as "previous"), then resume the running-sum rule from there until the next boundary.
Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
