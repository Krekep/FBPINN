from abc import ABC
from typing import Callable
import torch


def RMSE(y_true, y_pred):
    """
    This class provides Root Mean squared Error loss function:
    $$ RMSE = \sqrt{MSE} $$
    """
    loss = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    return loss


# Reduction should be set to None?
_losses: dict = {
    "Huber": torch.nn.HuberLoss(),
    "MeanAbsoluteError": torch.nn.L1Loss(),
    "MeanSquaredError": torch.nn.MSELoss(),
    "MSE": torch.nn.MSELoss(),
    "RootMeanSquaredError": RMSE,
    "RMSE": RMSE,
}


def get_loss(name: str):
    """
    Get loss function by name
    Parameters
    ----------
    name: str
        Name of loss function

    Returns
    -------
    loss_class: tf.keras.losses.Loss
        Result loss function
    """
    return _losses.get(name)


def get_all_loss_functions() -> dict[str, torch.nn.Module]:
    """
    Get all loss functions
    Parameters
    ----------

    Returns
    -------
    loss_class: dict[str, tf.keras.losses.Loss]
        All loss functions
    """
    return _losses
