"""Real English text, one-step dependency: write the previous letter at each position (the first
position has none and gets #). Windows of WikiText-2 paragraphs reduced to letters. Added after the
ascending and vowel-count rules proved to be at chance without markers; the rule is that of
tasks/copy_previous."""
from tasks.common.boundary import insert_boundaries
from tasks.common.real_data import window

TASK_NAME = "text_copy_previous"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [16, 32, 64]


def rule(s):
    return "#" + s[:-1]


assert rule("thelazydog") == "#thelazydo"


def build_prompt(N, k, rng, axis_val=None):
    s = window("wikitext_letters", N, rng)
    out = rule(s)
    inp_b, out_b = insert_boundaries(s, k), insert_boundaries(out, k)
    prompt = f"""You are a deterministic sequence transformer.
Task: Copy the previous letter.
Alphabet: lowercase letters a-z in the input (English text with spaces and punctuation removed); the same letters and # in the output.
Rule: At each position, output the letter that comes immediately before it in the input. The first position has no previous letter and gets #.
Example: Input thelazydog → Output #thelazydo
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {inp_b}
Output: """
    return inp_b, out_b, prompt
