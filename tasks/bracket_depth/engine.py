"""Naturalistic prefix-dependent task (reviewer question): nesting depth of a parenthesis string.
output[i] = (number of '(' minus number of ')' among input[0..i]) mod 4, written as a digit. Every
position depends on the whole prefix, so a marker boundary always separates needed state."""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "bracket_depth"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
MOD = 4


def _depth(s):
    out, d = [], 0
    for ch in s:
        d += 1 if ch == "(" else -1
        out.append(str(d % MOD))
    return "".join(out)


def build_prompt(N, k, rng, axis_val=None):
    input_str_base = "".join(rng.choice("()") for _ in range(N))
    output_str_base = _depth(input_str_base)
    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)
    prompt = f"""You are a deterministic sequence transformer.
Task: Parenthesis nesting depth.
Alphabet: ( and ) in the input; digits 0-{MOD - 1} in the output.
Rule: Read the input left to right, keeping a running depth that starts at 0. Each ( adds 1 and each ) subtracts 1. After each character, output the current depth modulo {MOD} (so a depth of -1 is written as {(-1) % MOD}, -2 as {(-2) % MOD}).
Example: Input (()) -> Output 1210 (depths 1, 2, 1, 0)
Example: Input )(( -> Output 301 (depths -1 -> 3, 0, 1)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """
    return input_bounded, output_bounded, prompt
