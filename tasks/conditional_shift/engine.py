"""Tier 1 (position-local): branching substitution cipher, with an
Alphabet_Size sweep {2,4,8} -- matches the other Tier-1 tasks' axis.

Why this task exists: cyclic_shift, character_substitution, and
random_permutation are all a single flat rule/lookup applied to every
character. This task requires classifying the character FIRST (is its own
alphabet index even or odd?) and THEN applying a different rule depending
on the class -- and unlike character_substitution/random_permutation, the
prompt does NOT hand over a flattened lookup table, so the model must
actually apply the classify-then-shift logic itself rather than doing a
pure retrieval. Still strictly position-local: the branch condition is a
property of the character itself, never its position in the sequence --
output[i] depends only on input[i].

Rule: characters at an EVEN alphabet index shift forward by 1; characters
at an ODD alphabet index shift backward by 1. This pairs up every adjacent
(2i, 2i+1) pair and swaps them -- verified computationally to be a valid
bijection with zero fixed points at every alphabet size. Deliberately
simple to state (unlike a within-class-rotation design that was tried and
discarded for being awkward to phrase without ambiguity) while still
requiring real per-character classification, not lookup.

size=2 necessarily coincides with cyclic_shift / character_substitution /
random_permutation at that one size (only one non-identity permutation
exists on a 2-element set) -- same documented degeneracy as the other
Tier-1 tasks.

Boundary insertion goes through tasks.common.boundary.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "conditional_shift"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True

ALPHABETS = {
    2: "AB",
    4: "ABCD",
    8: "ABCDEFGH",
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
