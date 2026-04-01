from typing import Callable

import torch
import keras


def perceptron_threshold(x, threshold: float = 1.0):
    return keras.ops.where(x >= threshold, 1.0, 0.0)


def sin_act(x):
    return keras.ops.sin(x)


def parabolic(x: torch.Tensor, beta: float = 0, p: float = 1 / 5):
    """
    Activation function is described in https://rairi.frccsc.ru/en/publications/426

    Parameters
    ----------
    x: tf.Tensor
        Input data vector
    beta: float
        Offset along the OY axis
    p: float
        Focal parabola parameter

    Returns
    -------
    new_x: tf.Tensor
        Data vector after applying activation function
    """
    return keras.ops.where(x >= 0.0, beta + keras.ops.sqrt(2.0 * p * x), beta - keras.ops.sqrt(-2.0 * p * x))


_activation_name = {
    "elu": keras.activations.elu,
    "relu": keras.activations.relu,
    "gelu": keras.activations.gelu,
    "selu": keras.activations.selu,
    "exponential": keras.activations.exponential,
    "linear": torch.nn.Identity(),
    "sigmoid": torch.nn.Sigmoid(),
    "hard_sigmoid": torch.nn.Hardsigmoid(),
    "swish": torch.nn.SiLU(),
    "tanh": torch.nn.Tanh(),
    "softplus": keras.activations.softplus,
    "softsign": keras.activations.softsign,
    "parabolic": parabolic,
    "sin": sin_act
}


def get_activation(name: str) -> Callable:
    """
    Get activation function by name
    Parameters
    ----------
    name: str
        name of activation function
    Returns
    -------
    func: Callable
        activation function
    """
    return _activation_name[name]


def get_all_activations() -> dict[str, Callable]:
    """
    Get all activation functions
    Parameters
    ----------

    Returns
    -------
    func: dict[str, Callable]
        dictionary of activation functions
    """
    return _activation_name
