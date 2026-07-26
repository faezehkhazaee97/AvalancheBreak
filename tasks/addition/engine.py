"""Tier 3 (global state): integer addition of two N-digit base-D numbers,
direction-misaligned (carries flow right-to-left, opposite generation
direction). Digit_Max sweep -- but {4, 8} only, NOT {2, 4, 8}.

Digit_Max=2 (binary addition) is excluded on purpose, not a gap to fill in:
the sampler forces both numbers' leading digit non-zero, which in base 2
means both leading digits are forced to 1, so both numbers are >= 2^(N-1)
and their sum is *always* >= 2^N -- i.e. addition always overflows N digits.
"No overflow" and "no leading zero" are mutually exclusive for base-2
N-digit addition; there is no valid non-degenerate trial to sample, not a
rare rejection-sampling miss. (The old sweep script asked for Digit_Max=2
anyway; its 200-attempt rejection loop always failed and the exception was
silently swallowed, so those rows just never appeared in the CSV.)

Boundaries are inserted within each operand SEPARATELY and only then joined
with "+" -- this is the fix for the bug where the paper's original headline
addition engine (AdditionTaskEngine in run_addition_templated_v2.py) built
`list(a_str) + ["+"] + list(b_str)` and counted straight through the "+"
character, so boundaries could land adjacent to or straddling the operator
(e.g. real example: "5050|+111|5"). Verified fix on Qwen2.5-7B: the
qualitative "boundaries hurt addition" finding survives, but the bug had
been overstating the effect (N=4 Cellular_Bounds TAR: 0.183 buggy vs 0.535
fixed).
"""

from tasks.common.boundary import insert_boundaries

TASK_NAME = "addition"
AXIS_NAME = "Digit_Max"
AXIS_VALUES = [4, 8]
OUTPUT_HAS_BOUNDARIES = False


def _sample_pair(N, digit_max, rng):
    for _ in range(200):
        a = [rng.randrange(digit_max) for _ in range(N)]
        b = [rng.randrange(digit_max) for _ in range(N)]
        if a[0] == 0:
            a[0] = rng.randrange(1, digit_max)
        if b[0] == 0:
            b[0] = rng.randrange(1, digit_max)
        carry = 0
        total = []
        for i in range(N - 1, -1, -1):
            s = a[i] + b[i] + carry
            total.append(s % digit_max)
            carry = s // digit_max
        if carry == 0:
            return a, b, list(reversed(total))
    raise RuntimeError(f"failed to sample a non-overflowing {N}-digit base-{digit_max} pair")


def build_prompt(N, k, rng, axis_val):
    digit_max = axis_val
    a, b, total = _sample_pair(N, digit_max, rng)

    a_str = "".join(map(str, a))
    b_str = "".join(map(str, b))
    sum_str = "".join(map(str, total))

    a_bounded = insert_boundaries(a_str, k)
    b_bounded = insert_boundaries(b_str, k)
    input_str = f"{a_bounded}+{b_bounded}"

    prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {N}-digit numbers in base {digit_max}.
Constraint: The input may contain vertical bars (|) as structural boundaries.
Ignore them when computing. The + symbol is the operator, ignored when summing.
Output ONLY the {N}-digit sum, zero-padded. No spaces, no explanations.

Input: {input_str}
Output: """

    return input_str, sum_str, prompt
