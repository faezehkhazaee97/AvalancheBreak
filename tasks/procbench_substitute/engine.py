"""External Tier-1 (position-local) task, sourced from a real third-party
benchmark: ProcBench's "Substitute" task (task06 in the HF dataset
ifujisawa/procbench, arXiv:2410.03117) -- "Replace characters in a given
string according to a given list of character pairs."

Uses ProcBench's OWN reference generator (vendored, MIT-licensed, in
tasks/common/procbench_vendor/generators.py -- GeneratorSubstitute), not a
reimplementation, following the same principle already used for CLRS-Text.

Selection rationale (see full 23-task audit in project notes): of
ProcBench's 23 task types, this was the ONLY one that is simultaneously
(a) a string task, (b) length-preserving, and (c) genuinely per-position
(output[i] is a pure function of input[i] via table lookup, verified by
hand against real generated examples -- no chaining, since generated
tables never have a key that is also a value). Every other candidate
either changes length, requires a global operation (sort, search,
rotation), or turned out on inspection to not be per-position after all
(task11's rules can map one input character to a TWO-character output,
which breaks length-preservation).

Structural note: ProcBench's native alphabet is lowercase letters + digits
(36 symbols), and by construction num_substitutions == length (every
position is substituted, not a partial table) -- both preserved as-is
rather than forced into our own alphabet-size convention, matching how
CLRS-Text's native float alphabet was kept rather than adapted.
"""

import os
import sys

_TASKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDOR_DIR = os.path.join(_TASKS_DIR, "common", "procbench_vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
if _TASKS_DIR not in sys.path and os.path.dirname(_TASKS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_TASKS_DIR))

from generators import GeneratorSubstitute  # noqa: E402

from tasks.common.boundary import insert_boundaries  # noqa: E402

TASK_NAME = "procbench_substitute"
AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = [4, 8, 16, 32, 64]


def build_prompt(N, k, rng, axis_val=None):
    clrs_seed = rng.randint(0, 2**31 - 1)
    gen = GeneratorSubstitute(seed=clrs_seed)
    gen.set_config(length=N)
    states, config = gen.generate_single()
    input_str_base = states[0]
    output_str_base = states[-1]
    table = config["table"]

    input_bounded = insert_boundaries(input_str_base, k)
    output_bounded = insert_boundaries(output_str_base, k)

    table_str = ", ".join(f"{src}->{dst}" for src, dst in table.items())

    prompt = f"""You are a deterministic sequence transformer.
Task: Character substitution (ProcBench "Substitute" task).
Rule: Replace each character in the string according to this substitution table (characters not listed stay unchanged): {table_str}
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

    return input_bounded, output_bounded, prompt
