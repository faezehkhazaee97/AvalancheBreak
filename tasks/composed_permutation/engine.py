"""Tier 1 (position-local): TWO independent unstructured substitution
ciphers applied in sequence -- output[i] = mapping2[mapping1[input[i]]].
Alphabet_Size sweep {4, 8} only (size=2 excluded -- see below).

Why this task exists: random_permutation already removed the algebraic
shortcut from a single substitution (rotation/swap -> arbitrary lookup).
This task adds a second, independent lookup on top: the model must get
TWO arbitrary retrievals right per character, not one, so any single
lookup error propagates through to the final output. Still strictly
position-local -- output[i] depends only on input[i], never on other
positions -- but tests composition depth as a distinct difficulty axis
from table size.

Alphabet_Size=2 is EXCLUDED, not a gap: on a 2-element alphabet there is
exactly one non-identity permutation, and it is its own inverse, so
composing ANY two independent size-2 derangements always yields the
identity function -- mathematically forced, confirmed by direct
enumeration, not a rare sampling miss (same category of exclusion as
addition's Digit_Max=2).

Mapping generation: two independent fixed derangements per size, searched
(fixed seeds, not re-rolled at runtime) until: (a) neither mapping alone
nor their composition is a rotation, (b) the two mappings are not
identical to each other, (c) the composed mapping has zero fixed points.
All three conditions verified computationally before hardcoding below.

Boundary insertion goes through tasks.common.boundary.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "composed_permutation"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [4, 8]
OUTPUT_HAS_BOUNDARIES = True

ALPHABETS = {
    4: "ABCD",
    8: "ABCDEFGH",
}

MAPPING1 = {
    4: {"A": "B", "B": "A", "C": "D", "D": "C"},
    8: {"A": "B", "B": "A", "C": "G", "D": "E", "E": "D", "F": "C", "G": "H", "H": "F"},
}

MAPPING2 = {
    4: {"A": "D", "B": "C", "C": "A", "D": "B"},
    8: {"A": "C", "B": "G", "C": "E", "D": "H", "E": "B", "F": "D", "G": "F", "H": "A"},
}

MAPPINGS = {
    size: {c: MAPPING2[size][MAPPING1[size][c]] for c in ALPHABETS[size]}
    for size in [4, 8]
}
assert all(len(set(m.values())) == len(m) for m in MAPPINGS.values()), "composed mapping must be a bijection"
assert all(k != v for m in MAPPINGS.values() for k, v in m.items()), "composed mapping must have no fixed points"


def build_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    alphabet = ALPHABETS[alphabet_size]
    mapping1 = MAPPING1[alphabet_size]
    mapping2 = MAPPING2[alphabet_size]

    input_str_base = "".join(rng.choices(alphabet, k=N))
    output_str_base = "".join(mapping2[mapping1[c]] for c in input_str_base)

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    map1_str = ", ".join([f"{a}→{b}" for a, b in mapping1.items()])
    map2_str = ", ".join([f"{a}→{b}" for a, b in mapping2.items()])

    prompt = f"""You are a deterministic sequence transformer.
Task: Composed character substitution (apply two fixed mappings in sequence).
Alphabet: {", ".join(alphabet)}
Step 1 mapping: {map1_str}
Step 2 mapping: {map2_str}
Rule: For each character, first apply the Step 1 mapping, then apply the Step 2 mapping to that result.
This gives the final output character. Apply this to every character independently.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
