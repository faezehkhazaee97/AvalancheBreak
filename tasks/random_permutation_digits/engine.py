"""Digit-alphabet variant of random_permutation: the tokenization control.

Without markers, letter strings merge into multi-character tokens (48 letters -> 25-29
tokens on every model in the study), and a marker after each letter restores one token
per character; digits are one token per character either way (48 -> 48-49). Part of the
letter-task benefit can therefore be tokenization rather than the marker itself. This
task isolates that share.

Identical to tasks/random_permutation/engine.py except the symbols: digits 0-1 / 0-3 /
0-7 replace A-B / A-D / A-H, with the same fixed derangements (_PERMS), the same prompt
wording, and the same local-seed formula, so a given (N, k, axis_val, trial) draws the
same positions as the letter task, relabeled.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "random_permutation_digits"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True

ALPHABETS = {
    2: "01",
    4: "0123",
    8: "01234567",
}

# Same derangements as the letter task (index i maps to index _PERMS[size][i]).
_PERMS = {
    2: [1, 0],
    4: [3, 2, 0, 1],
    8: [4, 0, 1, 5, 6, 7, 2, 3],
}

MAPPINGS = {
    size: {alpha[i]: alpha[_PERMS[size][i]] for i in range(size)}
    for size, alpha in ALPHABETS.items()
}


def build_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    alphabet = ALPHABETS[alphabet_size]
    mapping = MAPPINGS[alphabet_size]

    input_str_base = "".join(rng.choices(alphabet, k=N))
    output_str_base = "".join(mapping[c] for c in input_str_base)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    mapping_str = ", ".join([f"{a}→{b}" for a, b in mapping.items()])

    prompt = f"""You are a deterministic sequence transformer.
Task: Character substitution mapping.
Alphabet: {", ".join(alphabet)}
Mapping: {mapping_str}
Rule: Replace each character with its mapped character. This is a fixed one-to-one substitution cipher.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
