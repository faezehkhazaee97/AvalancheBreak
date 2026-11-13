"""running_xor with the 'worked' marker instruction: the across wording plus two worked examples with
bars, the first annotated at the position right after its first bar (see tasks/common/wording.py)."""

from tasks.running_xor import engine as _base
from tasks.common.wording import make_worked

TASK_NAME = "running_xor_worked"
AXIS_NAME = _base.AXIS_NAME
AXIS_VALUES = _base.AXIS_VALUES
OUTPUT_HAS_BOUNDARIES = True
build_prompt = make_worked(_base)
