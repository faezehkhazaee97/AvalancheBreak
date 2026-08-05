"""Flip-flop language modeling (Liu et al. 2306.00946), LOW write-instruction
density -> long effective dependency (writes are rare, so a read/ignore
position often must carry register state across many positions since the
last write). See tasks/common/flip_flop_shared.py for the full task
definition.
"""

TASK_NAME = "flip_flop_low"
LENGTHS = [4, 8, 16, 32, 64]
WRITE_PROB = 0.05

from tasks.common.flip_flop_shared import (  # noqa: E402
    AXIS_NAME, AXIS_VALUES, OUTPUT_HAS_BOUNDARIES, make_build_prompt,
)

build_prompt = make_build_prompt(WRITE_PROB, "low")
