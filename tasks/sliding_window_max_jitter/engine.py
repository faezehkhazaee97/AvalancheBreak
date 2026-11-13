"""sliding_window_max with every marker moved by -1, 0 or +1 positions from its periodic place (k = 4 and 8;
see tasks/common/jitter.py). Inputs and seeds are those of the main grid's k = 4 and 8 cells, so
each jittered generation is paired with an exactly periodic one."""

from tasks.sliding_window_max import engine as _base
from tasks.common.jitter import make_jitter

TASK_NAME = "sliding_window_max_jitter"
AXIS_NAME = _base.AXIS_NAME
AXIS_VALUES = _base.AXIS_VALUES
OUTPUT_HAS_BOUNDARIES = True
INTERVALS = [4, 8]
build_prompt = make_jitter(_base, amount=1)
