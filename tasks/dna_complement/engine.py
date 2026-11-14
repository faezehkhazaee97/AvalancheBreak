"""Real DNA, position-local: the complementary strand, base by base (A<->T, C<->G). Windows of a human
promoter sequence (tasks/common/real_data.py)."""
from tasks.common.boundary import insert_boundaries
from tasks.common.real_data import window

TASK_NAME = "dna_complement"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [16, 32, 64]
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def rule(s):
    return "".join(COMP[c] for c in s)


assert rule("ACGTTG") == "TGCAAC"


def build_prompt(N, k, rng, axis_val=None):
    s = window("promoters", N, rng)
    out = rule(s)
    inp_b, out_b = insert_boundaries(s, k), insert_boundaries(out, k)
    prompt = f"""You are a deterministic sequence transformer.
Task: DNA complement.
Alphabet: A, C, G, T (the input is a stretch of a human promoter sequence).
Rule: Replace every base by its complement: A becomes T, T becomes A, C becomes G, G becomes C. Each output base depends only on the input base at the same position.
Example: Input ACGTTG → Output TGCAAC
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {inp_b}
Output: """
    return inp_b, out_b, prompt
