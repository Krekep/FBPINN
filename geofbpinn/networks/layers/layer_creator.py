from collections import defaultdict
from typing import Type

import torch

from geofbpinn.networks.layers.dense import DenseLayer


def create(
    inp_size,
    shape,
    activation="linear",
    weight=torch.nn.init.ones_,
    bias=torch.nn.init.zeros_,
    layer_type="Dense",
    is_debug=False,
    device=None,
    **kwargs
) -> torch.nn.Module:
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
        inp_size, shape, activation, weight, bias, is_debug=is_debug, device=device, **kwargs  # type: ignore
    )
    return layer


_create_functions: defaultdict[str, Type[torch.nn.Module]] = defaultdict(
    lambda: DenseLayer
)
_create_functions["Dense"] = DenseLayer
