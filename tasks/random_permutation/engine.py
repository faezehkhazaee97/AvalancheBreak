"""Tier 1 (position-local): fixed one-to-one substitution cipher using an
UNSTRUCTURED random permutation, with an Alphabet_Size sweep {2,4,8} --
matches cyclic_shift/character_substitution's axis for direct comparison.

Why this task exists: cyclic_shift (rotate-by-k) and character_substitution
(swap-by-half-alphabet) are both algebraically regular mappings -- modular
arithmetic on the character's index, which a model can compute with a
learned "shift" circuit rather than truly retrieving each character's value
from the table given in the prompt. This task removes that shortcut: the
per-size mapping is a fixed, randomly generated derangement (no character
maps to itself) with no closed-form rule, verified computationally to NOT
be a rotation for size=4 and size=8 (unavoidable at size=2: a 2-element set
has exactly one non-identity permutation, and it coincides with rotation-by-1
by necessity -- same caveat character_substitution already documents).

Mapping generation: random.Random(20260715) [fixed seed, independent of the
per-trial seed] -> reject-sample derangements until no fixed points ->
verified size=4 and size=8 are not equal to any rotation-by-k.

Boundary insertion goes through tasks.common.boundary (avoids the k>=N
degenerate-cell bug from the start).
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "random_permutation"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True

ALPHABETS = {
    2: "AB",
    4: "ABCD",
    8: "ABCDEFGH",
}

# fixed, verified-non-rotational (for size>=4) random derangements
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
