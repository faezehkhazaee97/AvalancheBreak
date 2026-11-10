"""running_xor with the 'across' marker instruction (see tasks/common/wording.py)."""

from tasks.running_xor import engine as _base
from tasks.common.wording import make_wording

TASK_NAME = "running_xor_across"
AXIS_NAME = _base.AXIS_NAME
AXIS_VALUES = _base.AXIS_VALUES
OUTPUT_HAS_BOUNDARIES = True
build_prompt = make_wording(_base, "across")
