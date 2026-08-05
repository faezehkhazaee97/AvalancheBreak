"""Flip-flop language modeling (Liu et al. 2306.00946), HIGH write-instruction
density -> short effective dependency (writes are frequent, so a read/ignore
position rarely needs to look far back for the last write). See
tasks/common/flip_flop_shared.py for the full task definition.
"""

TASK_NAME = "flip_flop_high"
LENGTHS = [4, 8, 16, 32, 64]
WRITE_PROB = 0.4

from tasks.common.flip_flop_shared import (  # noqa: E402
    AXIS_NAME, AXIS_VALUES, OUTPUT_HAS_BOUNDARIES, make_build_prompt,
)

build_prompt = make_build_prompt(WRITE_PROB, "high")
