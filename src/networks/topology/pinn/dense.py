from typing import List, Optional, Dict, Callable

import keras
import torch

from src.geometry.decomposition import Block
from src.networks import layer_creator, losses, metrics, optimizers


class PhysicsInformedNet(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        block_size: list = None,
        output_size: int = 10,
        activation_func: str = "linear",
        weight=keras.initializers.RandomUniform(minval=-1, maxval=1),
        biases=keras.initializers.RandomUniform(minval=-1, maxval=1),
        layer: str | List[str] = "Dense",
        is_debug: bool = False,
        name="",
        device=None,
        **kwargs,
    ):
        self._name = "PINN" if name == "" else name
        decorator_params: List[Optional[Dict]] = [None]
        if "decorator_params" in kwargs.keys():
            decorator_params = kwargs.get("decorator_params")
            kwargs.pop("decorator_params")
        else:
            decorator_params = [None]

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is None
            or decorator_params is None
        ):
            decorator_params = [None] * (len(block_size) + 1)

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is not None
        ):
            decorator_params = decorator_params * (len(block_size) + 1)

        self.block: Block = kwargs["domain_block"]
        kwargs.pop("domain_block")
        if "layers" in kwargs.keys():
            self.blocks: List[torch.nn.Module] = kwargs["layers"][:-1]
            self.out_layer: torch.nn.Module = kwargs["layers"][-1]
            kwargs.pop("layers")
            super(PhysicsInformedNet, self).__init__(**kwargs)
        else:
            super(PhysicsInformedNet, self).__init__(**kwargs)
            layers: List[torch.nn.Module] = []

            if not isinstance(activation_func, list):
                activation_func = [activation_func] * (len(block_size) + 1)
            if not isinstance(layer, list):
                layer = [layer] * (len(block_size) + 1)
            block_size = [input_size] + block_size
            for i in range(1, len(block_size)):
                layers.append(
                    layer_creator.create(
                        block_size[i - 1],
                        block_size[i],
                        activation=activation_func[i],
                        weight=weight,
                        bias=biases,
                        layer_type=layer[i],
                        is_debug=is_debug,
                        name=f"Layer{i}",
                        decorator_params=decorator_params[i],
                        device=device
                    )
                )
            last_block_size = block_size[-1]
            self.blocks = torch.nn.ModuleList(layers)

            self.out_layer = layer_creator.create(
                last_block_size,
                output_size,
                activation=activation_func[-1],
                weight=lambda x: torch.nn.init.xavier_uniform_(x, gain=0.01),
                bias=biases,
                layer_type=layer[-1],
                is_debug=is_debug,
                name=f"OutLayer",
                decorator_params=decorator_params[-1],
                device=device
            )

        self.activation_funcs = activation_func
        self.weight_initializer = weight
        self.bias_initializer = biases
        self.input_size = input_size
        self.block_size = block_size
        self.output_size = output_size
        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}

    @property
    def get_activations(self) -> List:
        """
        Get list of activations functions for each layer

        Returns
        -------
        activation: list
        """
        return [layer.get_activation for layer in self.blocks]

    def call(self, inputs, **kwargs):
        """
        Obtaining a neural network response on the input data vector
        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """
        x = inputs
        for layer in self.blocks:
            x = layer(x, **kwargs)
        return self.out_layer(x, **kwargs)

    def forward(self, inputs, **kwargs):
        return self.call(inputs, **kwargs)

