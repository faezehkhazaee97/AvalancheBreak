"""Real English text, one-step dependency: 1 if the letter comes later in the alphabet than the
previous letter, else 0; the first position is 0. Windows of WikiText-2 paragraphs (letters only)."""
from tasks.common.boundary import insert_boundaries
from tasks.common.real_data import window

TASK_NAME = "text_ascending"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [16, 32, 64]


def rule(s):
    return "0" + "".join("1" if s[i] > s[i - 1] else "0" for i in range(1, len(s)))


assert rule("thelazydog") == "0010101001"


def build_prompt(N, k, rng, axis_val=None):
    s = window("wikitext_letters", N, rng)
    out = rule(s)
    inp_b, out_b = insert_boundaries(s, k), insert_boundaries(out, k)
    prompt = f"""You are a deterministic sequence transformer.
Task: Later in the alphabet than the previous letter.
Alphabet: lowercase letters a-z in the input (English text with spaces and punctuation removed); digits 0 and 1 in the output.
Rule: For each letter, output 1 if it comes later in the alphabet than the letter immediately before it, and 0 otherwise (also 0 if the two letters are the same). The first letter has no previous letter and gets 0.
Example: Input thelazydog → Output 0010101001 (t: first, 0; h after t: 0; e after h: 0; l after e: 1; a after l: 0; z after a: 1; y after z: 0; d after y: 0; o after d: 1; g after o: 0)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {inp_b}
Output: """
    return inp_b, out_b, prompt
