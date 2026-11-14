"""Real English text, prefix state: running count of vowels (a, e, i, o, u) so far, modulo 4.
Windows of WikiText-2 paragraphs (letters only)."""
from tasks.common.boundary import insert_boundaries
from tasks.common.real_data import window

TASK_NAME = "text_vowel_count"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [16, 32, 64]
MOD = 4


def rule(s):
    out, c = [], 0
    for ch in s:
        c += ch in "aeiou"
        out.append(str(c % MOD))
    return "".join(out)


assert rule("thelazydog") == "0011222233"


def build_prompt(N, k, rng, axis_val=None):
    s = window("wikitext_letters", N, rng)
    out = rule(s)
    inp_b, out_b = insert_boundaries(s, k), insert_boundaries(out, k)
    prompt = f"""You are a deterministic sequence transformer.
Task: Running vowel count modulo {MOD}.
Alphabet: lowercase letters a-z in the input (English text with spaces and punctuation removed); digits 0-{MOD - 1} in the output.
Rule: Read the input left to right, keeping a count of the vowels (a, e, i, o, u) seen so far, starting at 0. After each letter, output the count modulo {MOD}.
Example: Input thelazydog → Output 0011222233
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {inp_b}
Output: """
    return inp_b, out_b, prompt
