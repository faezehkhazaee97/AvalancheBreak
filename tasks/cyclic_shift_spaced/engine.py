"""Spaced-letter variant of cyclic_shift (tokenization control; see tasks/common/spaced.py).
Same seeds, letters, rule and prompt as tasks/cyclic_shift/engine.py; characters are separated by
spaces so every letter is one token with or without markers."""

from tasks.cyclic_shift import engine as _base
from tasks.common.spaced import make_spaced

TASK_NAME = "cyclic_shift_spaced"
AXIS_NAME = _base.AXIS_NAME
AXIS_VALUES = _base.AXIS_VALUES
OUTPUT_HAS_BOUNDARIES = True
build_prompt = make_spaced(_base)
