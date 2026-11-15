"""copy_previous with the 'neutral' marker instruction (see tasks/common/wording.py)."""

from tasks.copy_previous import engine as _base
from tasks.common.wording import make_wording

TASK_NAME = "copy_previous_neutral"
AXIS_NAME = _base.AXIS_NAME
AXIS_VALUES = _base.AXIS_VALUES
OUTPUT_HAS_BOUNDARIES = True
build_prompt = make_wording(_base, "neutral")
