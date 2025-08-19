from collections import defaultdict
from typing import Type

import numpy as np
from tensorflow import Tensor
from tensorflow import keras

from src.networks.layers.tf_dense import TensorflowDense


def create(
    inp_size,
    shape,
    activation="linear",
    weight=keras.initializers.get("ones"),
    bias=keras.initializers.get("zeros"),
    layer_type="Dense",
    is_debug=False,
    **kwargs
) -> keras.layers.Layer:
    """
    Create layer by parameters

    Parameters
    ----------
    inp_size: int
        layer input size
    shape: int
        amount of neurons in layer
    activation: str
        activation function for neurons
    weight
    bias
    layer_type: str
        type of layer for create
    is_debug: bool
    kwargs

    Returns
    -------
    layer
        Created layer
    """

    # mypy thinks the keras.layers.Layer constructor is being called,
    # so it complains about unknown arguments and a large number of arguments
    layer = _create_functions[layer_type](
        inp_size, shape, activation, weight, bias, is_debug=is_debug, **kwargs  # type: ignore
    )
    return layer


_create_functions: defaultdict[str, Type[keras.layers.Layer]] = defaultdict(
    lambda: TensorflowDense
)
_create_functions["Dense"] = TensorflowDense
