"""Minimal stub: clrs._src.probing only does `isinstance(element, tf.Tensor)`
to detect TF tensors before converting to numpy. We never pass real TF
tensors through this pipeline, so a dummy class that nothing will ever be
an instance of is a correct, side-effect-free substitute -- avoids a full
TensorFlow install just for one type check."""

class Tensor:
    pass
