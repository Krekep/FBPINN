from typing import Callable

import torch

_optimizers: dict = {
    "Adadelta": torch.optim.Adadelta,
    "Adafactor": torch.optim.Adafactor,
    "Adagrad": torch.optim.Adagrad,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "Adamax": torch.optim.Adamax,
    "Nadam": torch.optim.NAdam,
    "RMSprop": torch.optim.RMSprop,
    "SGD": torch.optim.SGD,
}


def get_optimizer(name: str):
    """
    Get optimizer by name
    Parameters
    ----------
    name: str
        Name of optimizer

    Returns
    -------
    optimizer_class: tf.keras.losses.Loss
        Result optimizer
    """
    return _optimizers.get(name)


def get_all_optimizers() -> dict[str, Callable]:
    """
    Get all optimizers
    Parameters
    ----------

    Returns
    -------
    optimizer_class: dict[str, tf.keras.losses.Loss]
        All optimizers
    """
    return _optimizers
