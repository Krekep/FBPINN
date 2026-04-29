from abc import ABC
from typing import Callable

import torch

from geofbpinn.networks.losses import get_all_loss_functions


_metrics: dict = {}

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
    metric_class: torch.nn.Module
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
    metric_class: dict[str, torch.nn.Module]
        All metrics
    """
    return _metrics
