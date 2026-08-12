"""CLRS-Text 'find_maximum_subarray_kadane' algorithm (Markeeva et al.,
arXiv:2406.04229) as a Tier-2-analog task: Kadane's algorithm, tracking the
(start, end) indices of the best subarray ending at or before each position.
A genuine external, third-party benchmark task -- selected because its
recurrence (best-subarray-so-far, updated from the immediately preceding
state) is the same one-step-back dependency shape as our own
running_sum_local/running_xor.

Uses CLRS-Text's own reference generator (vendored in
tasks/common/clrs_vendor/, from https://github.com/google-deepmind/clrs,
Apache 2.0) -- real (signed) arrays and real ground truth from their code.

Same token-level infrastructure as clrs_minimum/engine.py (see that file's
docstring for the full rationale) via SCORING_MODE = "token". One extra
formatting step here: CLRS-Text's own tuple format is "(2, 2)" with a space
after the inner comma, which would collide with ", " as the outer
token-list separator (naive splitting would cut a tuple in half). Tuples
are compacted to "(2,2)" (no inner space) before being treated as atomic
tokens, so ", " stays unambiguous as the sole separator everywhere in this
study's token-mode tasks.

N here means TRACE length (positions 1..N reported); the underlying CLRS
array has N+1 elements, same convention and same empirically-verified
"final answer = position N's state under the same rule" property as
clrs_minimum.
"""

import os
import sys

_TASKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDOR_DIR = os.path.join(_TASKS_DIR, "common", "clrs_vendor")
_TF_STUB_DIR = os.path.join(_VENDOR_DIR, "tf_stub")
for _p in (_TF_STUB_DIR, _VENDOR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _TASKS_DIR not in sys.path and os.path.dirname(_TASKS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_TASKS_DIR))

from clrs._src.clrs_text.huggingface_generators import clrs_generator  # noqa: E402
from tasks.common.boundary import insert_boundaries_tokens  # noqa: E402

TASK_NAME = "clrs_kadane"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
SCORING_MODE = "token"
TOKEN_REGEX = r"\([^)]*\)"
LENGTHS = [4, 8, 16, 32, 64, 128]


def estimate_max_new_tokens(N):
    return N * 10 + 40


def _compact(tuple_str):
    """'(2, 2)' -> '(2,2)' -- removes the inner-comma space so tuples stay
    atomic under ", "-splitting."""
    return tuple_str.replace(", ", ",")


def _parse_example(question, answer):
    key_part = question.split("key: [")[1].split("]")[0]
    array_vals = key_part.split()
    initial_tuple = question.split("initial_trace: ")[1].split("\n")[0].strip()
    trace_part, final_part = answer.split("|")
    trace_part = trace_part.strip()
    if trace_part.endswith(","):
        trace_part = trace_part[:-1].strip()
    # split on "), (" to isolate tuples, re-adding the stripped parens
    raw_tuples = trace_part.split("), (")
    trace_tuples = []
    for i, t in enumerate(raw_tuples):
        t = t.strip()
        if not t.startswith("("):
            t = "(" + t
        if not t.endswith(")"):
            t = t + ")"
        trace_tuples.append(_compact(t))
    final_tuple = _compact(final_part.strip())
    return array_vals, _compact(initial_tuple), trace_tuples, final_tuple


def build_prompt(N, k, rng, axis_val=None):
    array_size = N + 1
    clrs_seed = rng.randint(0, 2**31 - 1)
    gen = clrs_generator({"find_maximum_subarray_kadane": [array_size]}, num_samples=1, use_hints=True, seed=clrs_seed)
    ex = next(iter(gen))
    array_vals, initial_tuple, trace_tuples, final_tuple = _parse_example(ex["question"], ex["answer"])

    output_tokens = trace_tuples + [final_tuple]
    assert len(output_tokens) == N, f"expected {N} output tokens, got {len(output_tokens)}: {output_tokens}"
    assert len(array_vals) == array_size, f"expected {array_size} array values, got {len(array_vals)}"

    array_bounded = insert_boundaries_tokens(array_vals, k)
    output_bounded = insert_boundaries_tokens(output_tokens, k)

    input_str = ", ".join(array_bounded)
    truth_str = ", ".join(output_bounded)

    prompt = f"""You are a deterministic sequence transformer, running Kadane's algorithm (find_maximum_subarray_kadane) from the CLRS-Text benchmark (a real textbook algorithm, tracking the best contiguous subarray).
Task: at each position, track (start,end) -- the index range of the highest-sum contiguous subarray found so far, scanning left to right. Values can be negative.
Array (positions 0 to {N}, comma-separated): {input_str}
Position 0's (start,end) is given: {initial_tuple} (only one value seen, trivially itself).
Rule: at each position i from 1 to {N}, either extend the current best-subarray-ending-at-i (if that beats starting fresh at i) or the running best (start,end) stays the same as the previous position's answer, updating only when a new candidate subarray beats it. Output the current best (start,end) at every position -- write each pair compactly as "(start,end)" with no space after the comma inside the parentheses.
IMPORTANT: The array and the expected output both contain vertical bars (|), structural boundaries. Copy them through unchanged, at the same relative position.
No explanations, no extra text -- output ONLY the comma-separated list of (start,end) pairs (and boundary bars).

Position 0: {initial_tuple}
Output (positions 1-{N}): """

    return input_str, truth_str, prompt
