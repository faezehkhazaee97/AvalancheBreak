"""running_sum_local at the spacings k = 5 and 6, which the main grid does not include (reviewer question:
does the blind spot depend on the spacing continuously?). Same prompts, inputs and seeds as the
base engine; the no-marker reference is the main grid's."""

from tasks.running_sum_local import engine as _base

TASK_NAME = "running_sum_local_finek"
AXIS_NAME = _base.AXIS_NAME
AXIS_VALUES = _base.AXIS_VALUES
OUTPUT_HAS_BOUNDARIES = True
INTERVALS = [5, 6]
build_prompt = _base.build_prompt
