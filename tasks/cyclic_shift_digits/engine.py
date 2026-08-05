"""Digit-alphabet variant of cyclic_shift, built to quantify the
tokenization confound (results_clean/_diagnostics/alphabet_tokenization_
audit.csv): digit alphabets tokenize 1:1 on every documented model, so this
task carries none of the multi-character-token merging that inflates the
letter-alphabet Tier-1 benefit at Base. Identical grid, seeds, and rule to
tasks/cyclic_shift/engine.py -- only the alphabet symbols change (0-1/0-3/
0-7 digits instead of A-B/A-D/A-H letters), so the same local-seed formula
(90_000*N + 100*k + 10*axis_val + t) draws the same rng.choices() sequence
positionally, just relabeled. The letter-vs-digit gap at matched (N, k,
axis_val) directly measures how much of the Tier-1 gain is atomicity
(tokenization) vs. attention scaffolding.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "cyclic_shift_digits"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [4, 8, 16, 32, 64, 128, 256]


def build_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    alphabet = [str(i) for i in range(alphabet_size)]

    input_str_base = "".join(rng.choices(alphabet, k=N))
    output_str_base = "".join(
        alphabet[(alphabet.index(c) + 1) % alphabet_size] for c in input_str_base
    )

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic digit shift.
Alphabet: {', '.join(alphabet)}
Rule: Replace every digit with the next one in the alphabet.
Mapping: {alphabet[-1]} -> {alphabet[0]}, and {alphabet[0]} -> {alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
