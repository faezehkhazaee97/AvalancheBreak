"""External Tier-1 (position-local) task: plain string copying, per Jelassi
et al. 2402.01032 ("Repeat After Me: Transformers are Better than State
Space Models at Copying"). output[i] = input[i] identically -- the
simplest possible per-position rule, and a useful floor/ceiling reference:
if boundaries help or hurt on literal copying, that isolates the effect
of structural markers from any actual per-character computation at all.

Procedurally generated from our own standard seeds (not sourced from an
external dataset -- the task definition itself, not example data, is what
comes from the paper), same alphabet-size sweep convention (2/4/8) as our
other Tier-1 tasks.
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "string_copy"
AXIS_NAME = "Alphabet_Size"
AXIS_VALUES = [2, 4, 8]
OUTPUT_HAS_BOUNDARIES = True


def build_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    alphabet = [chr(ord("A") + i) for i in range(alphabet_size)]
    input_str_base = "".join(rng.choice(alphabet) for _ in range(N))
    output_str_base = input_str_base  # identity

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    prompt = f"""You are a deterministic sequence transformer.
Task: Copy the string exactly.
Alphabet: {', '.join(alphabet)}
Rule: Output the input string unchanged, character for character.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
