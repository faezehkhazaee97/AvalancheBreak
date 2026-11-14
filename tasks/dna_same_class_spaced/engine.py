"""Spaced variant of dna_same_class (tokenization control; see tasks/common/spaced.py): same windows, rule and
prompt, characters separated by spaces so every character is one token."""
from tasks.dna_same_class import engine as _base
from tasks.common.spaced import make_spaced

TASK_NAME = "dna_same_class_spaced"
AXIS_NAME = _base.AXIS_NAME
AXIS_VALUES = _base.AXIS_VALUES
OUTPUT_HAS_BOUNDARIES = True
LENGTHS = _base.LENGTHS
build_prompt = make_spaced(_base)
