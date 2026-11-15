"""Real English text with its own delimiters (reviewer's question whether natural punctuation and word
breaks act like markers). Windows of WikiText-2 paragraphs in lowercase with spaces, commas, and periods
kept (data/real/wikitext_natural.txt, from scripts/make_wikitext_natural.py). The rule is
text_copy_previous over the letters only: each letter is replaced by the letter before it in the text,
skipping spaces and punctuation, the first letter gets #, and every space, comma, and period is copied
in place. The first letter after a space or punctuation mark therefore needs the letter on the other
side of it, as the position after a bar does. No bars are inserted (k = 0 only); N counts letters.
The letters-only comparison is text_copy_previous's no-marker run."""

import re
from functools import lru_cache
from tasks.common.real_data import DATA

TASK_NAME = "text_copy_previous_natural"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = False
LENGTHS = [16, 32, 64]
INTERVALS = [0]


@lru_cache(maxsize=None)
def _pool():
    lines = [l.strip() for l in (DATA / "wikitext_natural.txt").read_text().splitlines()]
    return [l for l in lines if sum(c.isalpha() for c in l) >= 64]


def window(N, rng):
    p = _pool()[rng.randrange(len(_pool()))]
    idx = [j for j, c in enumerate(p) if c.isalpha()]
    s = rng.randrange(len(idx) - N + 1)
    return p[idx[s]: idx[s + N - 1] + 1]


def rule(text):
    letters = [c for c in text if c.isalpha()]
    shifted = iter(["#"] + letters[:-1])
    return "".join(next(shifted) if c.isalpha() else c for c in text)


assert rule("the cat, sat.") == "#th eca, tsa."


def estimate_max_new_tokens(N):
    return 2 * N + 20


def build_prompt(N, k, rng, axis_val=None):
    s = window(N, rng)
    out = rule(s)
    prompt = f"""You are a deterministic sequence transformer.
Task: Copy the previous letter.
Alphabet: lowercase English text with spaces, commas, and periods; the output uses the same characters and #.
Rule: Replace each letter with the letter that comes before it in the text, skipping spaces and punctuation. The first letter has no previous letter and gets #. Copy every space, comma, and period unchanged at its place.
Example: Input the cat, sat. → Output #th eca, tsa.
No explanations.

Input: {s}
Output: """
    return s, out, prompt
