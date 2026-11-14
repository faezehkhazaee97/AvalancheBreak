"""Real DNA, one-step dependency: 1 if the base is in the same class as the previous base (purine A/G
or pyrimidine C/T), else 0; the first position is 0. Windows of a human promoter sequence."""
from tasks.common.boundary import insert_boundaries
from tasks.common.real_data import window

TASK_NAME = "dna_same_class"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [16, 32, 64]
PURINE = set("AG")


def rule(s):
    out = ["0"]
    for i in range(1, len(s)):
        out.append("1" if (s[i] in PURINE) == (s[i - 1] in PURINE) else "0")
    return "".join(out)


assert rule("AGCTTA") == "010110"


def build_prompt(N, k, rng, axis_val=None):
    s = window("promoters", N, rng)
    out = rule(s)
    inp_b, out_b = insert_boundaries(s, k), insert_boundaries(out, k)
    prompt = f"""You are a deterministic sequence transformer.
Task: Same base class as the previous base.
Alphabet: A, C, G, T in the input (a stretch of a human promoter sequence); digits 0 and 1 in the output.
Rule: A and G are purines; C and T are pyrimidines. For each base, output 1 if it is in the same class as the base immediately before it, and 0 otherwise. The first base has no previous base and gets 0.
Example: Input AGCTTA → Output 010110 (A: first, 0; G: purine after purine, 1; C: pyrimidine after purine, 0; T: pyrimidine after pyrimidine, 1; T: pyrimidine after pyrimidine, 1; A: purine after pyrimidine, 0)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {inp_b}
Output: """
    return inp_b, out_b, prompt
