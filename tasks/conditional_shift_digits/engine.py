"""Digit-alphabet variant of conditional_shift: the tokenization control.

Without markers, letter strings merge into multi-character tokens (48 letters -> 25-29
tokens on every model in the study), and a marker after each letter restores one token
per character; digits are one token per character either way. This task isolates how
much of the letter-task benefit is that tokenization effect.

Identical to tasks/conditional_shift/engine.py except the symbols: digits 0-1 / 0-3 /
0-7 replace A-B / A-D / A-H, with the same rule (even alphabet position -> next, odd ->
previous), the same prompt wording, and the same local-seed formula, so a given
(N, k, axis_val, trial) draws the same positions as the letter task, relabeled.

Known asymmetry: with digits in natural order, a symbol's alphabet position equals the
digit itself, so the even/odd classification is easier than for letters. Compare marker
effects (Bounds - Base) within each alphabet, not raw accuracy across alphabets.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "conditional_shift_digits"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True

ALPHABETS = {
    2: "01",
    4: "0123",
    8: "01234567",
}


def _build_mapping(size):
    alphabet = ALPHABETS[size]
    mapping = {}
    for i, c in enumerate(alphabet):
        mapping[c] = alphabet[(i + 1) % size] if i % 2 == 0 else alphabet[(i - 1) % size]
    return mapping


MAPPINGS = {size: _build_mapping(size) for size in [2, 4, 8]}
assert all(len(set(m.values())) == len(m) for m in MAPPINGS.values()), "mapping must be a bijection"
assert all(k != v for m in MAPPINGS.values() for k, v in m.items()), "mapping must have no fixed points"


def build_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    alphabet = ALPHABETS[alphabet_size]
    mapping = MAPPINGS[alphabet_size]

    input_str_base = "".join(rng.choices(alphabet, k=N))
    output_str_base = "".join(mapping[c] for c in input_str_base)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    example_even = alphabet[0]
    example_odd = alphabet[1]

    prompt = f"""You are a deterministic sequence transformer.
Task: Branching character shift.
Alphabet: {", ".join(alphabet)} (positions numbered 0, 1, 2, ... in this order)
Rule: For each character, look at its position number in the alphabet.
- If the position is EVEN, replace the character with the NEXT character in the alphabet (e.g. {example_even} -> {mapping[example_even]}).
- If the position is ODD, replace the character with the PREVIOUS character in the alphabet (e.g. {example_odd} -> {mapping[example_odd]}).
Wrap around the alphabet if needed.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
