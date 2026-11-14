"""Real English text, position-local: shift every letter one step forward in the alphabet (z wraps to
a). Windows of WikiText-2 paragraphs reduced to lowercase letters (tasks/common/real_data.py)."""
from tasks.common.boundary import insert_boundaries
from tasks.common.real_data import window

TASK_NAME = "text_shift"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [16, 32, 64]


def rule(s):
    return "".join(chr((ord(c) - 97 + 1) % 26 + 97) for c in s)


assert rule("thelazydog") == "uifmbazeph"


def build_prompt(N, k, rng, axis_val=None):
    s = window("wikitext_letters", N, rng)
    out = rule(s)
    inp_b, out_b = insert_boundaries(s, k), insert_boundaries(out, k)
    prompt = f"""You are a deterministic sequence transformer.
Task: Shift each letter forward by one.
Alphabet: lowercase letters a-z (the input is English text with spaces and punctuation removed).
Rule: Replace every letter by the next letter of the alphabet: a becomes b, b becomes c, ..., z becomes a. Each output letter depends only on the input letter at the same position.
Example: Input thelazydog → Output uifmbazeph
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {inp_b}
Output: """
    return inp_b, out_b, prompt
