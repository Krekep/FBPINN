from abc import ABC
from typing import Callable

import keras

from src.networks.losses import get_all_loss_functions


_metrics: dict = {
    "root_mean_squared_error": keras.metrics.RootMeanSquaredError(),
}

_metrics = dict(get_all_loss_functions(), **_metrics)


def get_metric(name: str):
    """
    Get metric by name
    Parameters
    ----------
    name: str
        Name of metric

    Returns
    -------
    metric_class: tf.keras.losses.Loss
        Result metric
    """
    return _metrics.get(name)


def get_all_metric_functions() -> dict[str, Callable]:
    """
    Get all metrics
    Parameters
    ----------

    Returns
    -------
    metric_class: dict[str, tf.keras.losses.Loss]
        All metrics
    """
    return _metrics
