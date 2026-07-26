"""running_max, but boundaries are real reset points instead of a pure
visual overlay. running_max's dependency spans the WHOLE prefix (output[i]
= max(input[0..i])), not just one step back, so "reset at boundary" means
something slightly different here: the running maximum restarts from
scratch within each boundary-delimited chunk, rather than carrying the
true running max across the boundary. This turns the task into a chunked
running-max -- each chunk computed independently of every other chunk.

See tasks/running_xor_reset/engine.py for the full rationale: this isolates
whether dense-boundary harm comes from breaking a still-needed dependency
(here: the running max carried from a previous chunk), versus boundary
insertion being disruptive on its own.

At k=0 (Base) this reduces to running_max exactly (one chunk = the whole
sequence).

Boundary-start position math: position j starts a new chunk iff j % k == 0
and j > 0, matching tasks.common.boundary.insert_boundaries.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "running_max_reset"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True


def _compute_running_max_reset(digit_str, k):
    n = len(digit_str)
    output = []
    running_max = -1
    for i, ch in enumerate(digit_str):
        is_chunk_start = k > 0 and i % k == 0 and i != 0
        if is_chunk_start:
            running_max = -1
        running_max = max(running_max, int(ch))
        output.append(str(running_max))
    return "".join(output)


def build_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    input_str_base = "".join(str(rng.randint(0, alphabet_size - 1)) for _ in range(N))
    output_str_base = _compute_running_max_reset(input_str_base, k)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Running maximum, reset at boundaries.
Alphabet: 0-{alphabet_size - 1}
Rule: At each position, output the LARGEST digit seen so far, including the current position.
Scan left to right; once a digit is output, it never decreases -- within the current block.
IMPORTANT: The input contains vertical bars (|), structural boundaries. Each boundary starts a brand new block: the running maximum RESETS to nothing at that point and starts counting fresh from the first digit of the new block, ignoring everything before the boundary.
Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
