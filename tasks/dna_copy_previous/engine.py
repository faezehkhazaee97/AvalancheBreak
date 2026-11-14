"""Real DNA, one-step dependency: write the previous base at each position (the first position has
none and gets N). Windows of a human promoter sequence (tasks/common/real_data.py). Added after the
same-class and GC-count rules proved to be at chance without markers, so that the real-DNA set has a
dependency task the models can do; the rule is that of tasks/copy_previous."""
from tasks.common.boundary import insert_boundaries
from tasks.common.real_data import window

TASK_NAME = "dna_copy_previous"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [16, 32, 64]


def rule(s):
    return "N" + s[:-1]


assert rule("ACGTTG") == "NACGTT"


def build_prompt(N, k, rng, axis_val=None):
    s = window("promoters", N, rng)
    out = rule(s)
    inp_b, out_b = insert_boundaries(s, k), insert_boundaries(out, k)
    prompt = f"""You are a deterministic sequence transformer.
Task: Copy the previous base.
Alphabet: A, C, G, T in the input (a stretch of a human promoter sequence); A, C, G, T and N in the output.
Rule: At each position, output the base that comes immediately before it in the input. The first position has no previous base and gets N.
Example: Input ACGTTG → Output NACGTT
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {inp_b}
Output: """
    return inp_b, out_b, prompt
