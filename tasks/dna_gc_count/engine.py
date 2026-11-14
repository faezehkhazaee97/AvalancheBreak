"""Real DNA, prefix state: running count of G and C bases so far, modulo 4. Windows of a human
promoter sequence."""
from tasks.common.boundary import insert_boundaries
from tasks.common.real_data import window

TASK_NAME = "dna_gc_count"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [16, 32, 64]
MOD = 4


def rule(s):
    out, c = [], 0
    for ch in s:
        c += ch in "GC"
        out.append(str(c % MOD))
    return "".join(out)


assert rule("ACGTGGC") == "0122301"


def build_prompt(N, k, rng, axis_val=None):
    s = window("promoters", N, rng)
    out = rule(s)
    inp_b, out_b = insert_boundaries(s, k), insert_boundaries(out, k)
    prompt = f"""You are a deterministic sequence transformer.
Task: Running GC count modulo {MOD}.
Alphabet: A, C, G, T in the input (a stretch of a human promoter sequence); digits 0-{MOD - 1} in the output.
Rule: Read the input left to right, keeping a count of the G and C bases seen so far, starting at 0. After each base, output the count modulo {MOD}.
Example: Input ACGTGGC → Output 0122301
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {inp_b}
Output: """
    return inp_b, out_b, prompt
