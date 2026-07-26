"""Tier 1 (position-local): fixed one-to-one substitution cipher, with an
Alphabet_Size sweep {2,4,8}.

The mapping is a half-alphabet-offset swap (pair letter i with letter
i + size/2), matching the spec doc's original design intent (A<->X, B<->Y,
... a 13-letter swap on the full 26-letter alphabet) scaled down to each
reduced alphabet size, e.g. for size=4: A<->C, B<->D.

This is a deliberate correction: the proven run_character_substitution_
alphabet_sweep.py script this repo actually used for the paper instead used
a rotate-by-1 mapping (A->B->C->D->A) for every alphabet size -- which is
mathematically IDENTICAL to cyclic_shift's own rule for the same alphabet
sizes (verified: {'A':'B','B':'C','C':'D','D':'A'} for both). That made
"character substitution" and "cyclic shift" the same transformation
function differing only in prompt wording (rule-stated vs. table-stated),
not two genuinely different tasks. The swap mapping here is not a rotation,
so it no longer collapses onto cyclic_shift's rule (except unavoidably at
size=2, where a 2-element alphabet has only one possible non-identity
permutation -- swap and rotate-by-1 are the same function by necessity,
there is no other non-trivial bijection to choose).

Boundary insertion goes through tasks.common.boundary (fixes the k>=N
degenerate-cell bug).
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "character_substitution"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True

ALPHABETS = {
    2: "AB",
    4: "ABCD",
    8: "ABCDEFGH",
}

MAPPINGS = {
    size: {
        alpha[i]: alpha[(i + size // 2) % size]
        for i in range(size)
    }
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
