"""Tier 1 (position-local): cyclic character shift, with an Alphabet_Size
sweep {2,4,8} -- this is the paper's actual headline task (Figure 1, the
SK-alpha fits, the error-recovery section's "N=8 through N=256").

Ported from run_core_templated_v2.py (internally codenamed "thermodynamics"
in the original repo -- an unrelated legacy name). NOT the spec doc's fixed
single-26-letter-alphabet version (run_cyclic_shift_structured_boundaries.py,
class CyclicShiftTaskEngine) -- that script's output CSV was never actually
generated and its data was never used anywhere in the paper. Shift wraps
within the reduced alphabet (e.g. alphabet_size=2: A<->B), not the full
26-letter alphabet. Sequence lengths extend to 256 (not just 64, unlike the
other 5 tasks) to match what the real headline data covers; boundary
insertion goes through tasks.common.boundary (fixes the k>=N degenerate-cell
bug -- the real script already filtered k<N by hand, so this changes
nothing for cyclic_shift specifically, just unifies the implementation).
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "cyclic_shift"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [4, 8, 16, 32, 64, 128, 256]


def build_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    alphabet = [chr(ord("A") + i) for i in range(alphabet_size)]

    input_str_base = "".join(rng.choices(alphabet, k=N))
    output_str_base = "".join(
        alphabet[(alphabet.index(c) + 1) % alphabet_size] for c in input_str_base
    )

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {alphabet[-1]} -> {alphabet[0]}, and {alphabet[0]} -> {alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
