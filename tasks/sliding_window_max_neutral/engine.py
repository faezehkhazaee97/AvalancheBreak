"""sliding_window_max with the 'neutral' marker instruction (see tasks/common/wording.py)."""

from tasks.sliding_window_max import engine as _base
from tasks.common.wording import make_wording

TASK_NAME = "sliding_window_max_neutral"
AXIS_NAME = _base.AXIS_NAME
AXIS_VALUES = _base.AXIS_VALUES
OUTPUT_HAS_BOUNDARIES = True
build_prompt = make_wording(_base, "neutral")
