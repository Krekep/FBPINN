from abc import ABC
from typing import Callable

import keras


def sign(x):
    return keras.ops.where(x < 0.0, -1.0, 1.0)


def relative_absolute_error(y_true, y_pred):
    """
    This class provides RAE loss function:
    $$ RAE = \frac{\Sum^n_{i=1} |y_i - \hat(y)_i|}{\Sum^n_{i=1} |y_i - \bar(y)|} $$
    """
    true_mean = keras.ops.mean(y_true)
    squared_error_num = keras.ops.sum(keras.ops.abs(y_true - y_pred))
    squared_error_den = keras.ops.sum(keras.ops.abs(y_true - true_mean))

    squared_error_den = keras.ops.cond(
        pred=keras.ops.equal(squared_error_den, 0.0),
        true_fn=lambda: 1.0,
        false_fn=lambda: squared_error_den,
    )

    loss = squared_error_num / squared_error_den
    return loss


def relative_l1_loss(y_true, y_pred):
    """
    This class provides Relative L1 loss function:
    $$ Loss = \frac{\frac{1}{n} * \Sum^n_{i=1} |y_i - \hat(y)_i|}{\frac{1}{n} * \Sum^n_{i=1} |y_i|}  $$
    """
    return keras.ops.mean(keras.ops.abs(y_true - y_pred)) / keras.ops.mean(keras.ops.abs(y_true))


def max_absolute_error(y_true, y_pred):
    """
    This class provides Max Absolute Deviation loss function:
    $$ MAD = \max |y - \hat(y)| $$
    """
    loss = keras.ops.max(keras.ops.abs(y_true - y_pred))
    return loss

def max_absolute_percentage_error(y_true, y_pred):
    """
    This class provides Max Absolute Percentage Error loss function:
    $$ MAD = \max |\frac{y - \hat(y)}{y}| $$
    """
    loss = keras.ops.max(keras.ops.abs((y_true - y_pred) / y_true)) * 100.0
    return loss


def RMSE(y_true, y_pred):
    """
    This class provides Root Mean squared Error loss function:
    $$ MAD = \sqrt{MSE} $$
    """
    loss = keras.ops.sqrt(keras.ops.mean((y_pred - y_true) ** 2))
    return loss


# Reduction should be set to None?
_losses: dict = {
    "Huber": keras.losses.Huber(),
    "LogCosh": keras.losses.LogCosh(),
    "MeanAbsoluteError": keras.losses.MeanAbsoluteError(),
    "MeanAbsolutePercentageError": keras.losses.MeanAbsolutePercentageError(),
    "MaxAbsolutePercentageError": max_absolute_percentage_error,
    "MeanSquaredError": keras.losses.MeanSquaredError(),
    "MSE": keras.losses.MeanSquaredError(),
    "RootMeanSquaredError": RMSE,
    "RMSE": RMSE,
    "MeanSquaredLogarithmicError": keras.losses.MeanSquaredLogarithmicError(),
    "RelativeAbsoluteError": relative_absolute_error,
    "MaxAbsoluteDeviation": max_absolute_error,
    "RelativeL1Loss": relative_l1_loss,
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


def get_all_loss_functions() -> dict[str, keras.losses.Loss]:
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
