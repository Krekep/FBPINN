from typing import Optional, List, Tuple, Dict

from keras import initializers
import torch
import torch.nn

from src.networks import activations


class DenseLayer(torch.nn.Module):
    """
    Class representing custom dense feed-forward layer
    """
    def __init__(
        self,
        input_dim: int = 32,
        units: int = 32,
        activation_func: str = "linear",
        weight_initializer=initializers.RandomNormal(),
        bias_initializer=initializers.Zeros(),
        is_debug: bool = False,
        name: str = "DenseLayer",
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        input_dim: int
            Input dimension size
        units: int
            Number of neurons in layer
        activation_func: str
            Name of activation function
        weight_initializer
            Random number generator for weight matrix
        bias_initializer
            Random number generator for bias vector
        is_debug: bool
            Add more output
        name: str
            Name of layer
        device: Optional[str]
            cpu/cuda/None. If None, then `cpu`
        """
        decorator_params = None

        if "decorator_params" in kwargs.keys():
            decorator_params = kwargs.get("decorator_params")
            kwargs.pop("decorator_params")

        if not isinstance(decorator_params, dict) and decorator_params is not None:
            raise TypeError(
                "Additional parameters for activation function must be dictionary"
            )

        if input_dim == 0 or units == 0:
            raise ValueError("Layer cannot have zero inputs or zero size")

        super(DenseLayer, self).__init__(**kwargs)
        self.w = torch.nn.Parameter(torch.randn(input_dim, units, device=device))
        self.b = torch.nn.Parameter(torch.randn(units, device=device))
        weight_initializer(self.w)
        bias_initializer(self.b)

        self.units = units
        self.input_dim = input_dim
        self._is_debug = is_debug
        self.activation_func = activations.get_activation(activation_func)
        self.activation_name = activation_func
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.decorator_params: Optional[dict] = decorator_params
        self._layer_name = name

    def forward(self, inputs, **kwargs):
        out = torch.matmul(inputs, self.w) + self.b
        if self.decorator_params is None:
            return self.activation_func(out)
        else:
            return self.activation_func(out, **self.decorator_params)

    def call(self, inputs, **kwargs):
        """
        Obtaining a layer response on the input data vector
        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """
        return self.forward(inputs, **kwargs)

    def __str__(self):
        res = f"Layer {self._layer_name}\n"
        res += f"weights shape = {self.w.shape}\n"
        if self._is_debug:
            # res += f"weights = {self.w.numpy()}\n"
            # res += f"biases = {self.b.numpy()}\n"
            res += f"activation = {self.activation_name}\n"
        return res
