"""Minimal init, patched locally: skip the GNN model stack (models/decoders/
processors/dataset/evaluation need chex/haiku/tensorflow), keep only the
sampler + spec + clrs_text machinery needed for text generation."""

from clrs._src import algorithms
from clrs._src import specs

from clrs._src.samplers import build_sampler
from clrs._src.samplers import CLRS30
from clrs._src.samplers import Features
from clrs._src.samplers import Feedback
from clrs._src.samplers import process_permutations
from clrs._src.samplers import process_pred_as_input
from clrs._src.samplers import process_random_pos
from clrs._src.samplers import Sampler
from clrs._src.samplers import Trajectory

from clrs._src.specs import ALGO_IDX_INPUT_NAME
from clrs._src.specs import CLRS_30_ALGS_SETTINGS
from clrs._src.specs import Location
from clrs._src.specs import OutputClass
from clrs._src.specs import Spec
from clrs._src.specs import SPECS
from clrs._src.specs import Stage
from clrs._src.specs import Type

__version__ = "2.0.3-patched"
