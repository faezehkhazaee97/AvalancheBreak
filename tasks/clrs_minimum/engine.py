"""CLRS-Text 'minimum' algorithm (Markeeva, McLeish, Ibarz, Bounsi, Kozlova,
Vitvitskyi, Blundell, Goldstein, Schwarzschild, Velickovic; arXiv:2406.04229)
as a Tier-3-analog task: track the index of the minimum value seen so far,
scanning left to right. A genuine external, third-party benchmark task, not
one we invented -- selected because its recurrence (argmin-so-far) is the
same full-prefix dependency shape as our own running_max, just tracking an
index instead of a value.

Uses CLRS-Text's own reference generator (vendored in
tasks/common/clrs_vendor/, from https://github.com/google-deepmind/clrs,
Apache 2.0) -- real arrays and real ground truth from their code, not a
reimplementation.

Structural differences from every other task in this study, both handled by
new shared infrastructure rather than bent to fit the old one:
  - CLRS's natural alphabet is floats, not a small discrete digit/letter
    set -- so there is no fixed AXIS_VALUES/valid_alphabet here.
  - Its natural output unit is a multi-character token per position (an
    index, potentially multi-digit), not a single character. Boundaries and
    scoring run at the TOKEN level (tasks.common.boundary.
    insert_boundaries_tokens, tasks.common.scoring.score_tokens) via
    SCORING_MODE = "token", which tasks.common.runner dispatches on.

N here means TRACE length (positions 1..N reported) -- the underlying CLRS
array has N+1 elements, since position 0's index (trivially itself) is
given in the prompt, matching the "first digit given, rest predicted"
convention used by every other task in this study. Verified empirically
(see session notes) that the CLRS trace + final-answer fields, concatenated,
give exactly N values following one uniform rule -- the "final answer" is
not a special case, it's just position N's state under the same recurrence,
reported in a separate field by CLRS-Text's own format.
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

TASK_NAME = "clrs_minimum"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
SCORING_MODE = "token"
LENGTHS = [4, 8, 16, 32, 64, 128]


def estimate_max_new_tokens(N):
    return N * 6 + 40


def _parse_example(question, answer):
    key_part = question.split("key: [")[1].split("]")[0]
    array_vals = key_part.split()
    initial_trace = question.split("initial_trace:")[1].split("\n")[0].strip()
    trace_part, final_part = answer.split("|")
    trace_tokens = [t.strip() for t in trace_part.strip().rstrip(",").split(",") if t.strip()]
    final_token = final_part.strip()
    return array_vals, initial_trace, trace_tokens, final_token


def build_prompt(N, k, rng, axis_val=None):
    array_size = N + 1
    clrs_seed = rng.randint(0, 2**31 - 1)
    gen = clrs_generator({"minimum": [array_size]}, num_samples=1, use_hints=True, seed=clrs_seed)
    ex = next(iter(gen))
    array_vals, initial_trace, trace_tokens, final_token = _parse_example(ex["question"], ex["answer"])

    output_tokens = trace_tokens + [final_token]
    assert len(output_tokens) == N, f"expected {N} output tokens, got {len(output_tokens)}: {output_tokens}"
    assert len(array_vals) == array_size, f"expected {array_size} array values, got {len(array_vals)}"

    array_bounded = insert_boundaries_tokens(array_vals, k)
    output_bounded = insert_boundaries_tokens(output_tokens, k)

    input_str = ", ".join(array_bounded)
    truth_str = ", ".join(output_bounded)

    prompt = f"""You are a deterministic sequence transformer, running the "minimum" algorithm from the CLRS-Text benchmark (a real array-scanning textbook algorithm).
Task: track the index of the smallest value seen so far, scanning the array left to right.
Array (positions 0 to {N}, comma-separated): {input_str}
Position 0's running-minimum index is given: {initial_trace} (only one value seen, trivially itself).
Rule: for each position i from 1 to {N}, compare array[i] to the value at the current running-minimum index. If array[i] is smaller, the running-minimum index becomes i; otherwise it stays exactly the same as the previous position's answer.
Output the running-minimum index at every position from 1 to {N}, as a comma-separated list.
IMPORTANT: The array and the expected output both contain vertical bars (|), structural boundaries. Copy them through unchanged, at the same relative position.
No explanations, no extra text -- output ONLY the comma-separated list of indices (and boundary bars).

Position 0 index: {initial_trace}
Output (positions 1-{N}): """

    return input_str, truth_str, prompt
