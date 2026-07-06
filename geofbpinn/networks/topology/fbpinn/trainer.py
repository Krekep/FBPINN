import torch

from geofbpinn.networks.topology.fbpinn.fbpinn_train import layer_train
from geofbpinn.networks.schedulers.layer import BaseLayerScheduler
from geofbpinn.networks.schedulers.loss import LossScheduler
from geofbpinn.networks.topology.fbpinn.model import FBPINN


def train_fbpinn(
    fbpinn: FBPINN,
    epochs: int,
    callbacks: list,
    val_input: torch.Tensor,
    val_truth: torch.Tensor,
    start_epoch: int = 0,
    patience: int = 300,
    log_interval: int = 100,
    eval_interval: int = 1,
    mode: str = "sequence",  # TODO: Rewrite to Enum
    png_salt: str = "",
    layer_scheduler: BaseLayerScheduler = None,
    loss_scheduler: LossScheduler = None,
    path_to_ckpt: str = "",
    lr_scheduler=None,
    configs: dict = None,
):
    """
    Interface for different training strategies.

    Parameters
    ----------
    fbpinn: FBPINN
        model for training
    epochs: int
        Number of epochs for training
    callbacks: list
    val_input: torch.Tensor
        Validation input
    val_truth: torch.Tensor
        Validation expected output
    start_epoch: int
        Training starts from this epoch
    patience: int
        Number of epochs before stopping training due to lack of improvement train loss
    log_interval: int
        Number of epoch between log results to console
    eval_interval: int
        Number of epoch between evaluation model on validation data
    mode: str
        Training strategy name
    png_salt: str
        Additional string to model name
    layer_scheduler: BaseLayerScheduler
        Scheduler for active submodels
    loss_scheduler: LossScheduler
        Scheduler for loss weights
    path_to_ckpt: str
        Path to checkpoint
    lr_scheduler:
        Learning rate scheduler
    configs: dict
        Model configs
    """
    if callbacks is None:
        callbacks = []

    for callback in callbacks:
        callback.set_model(fbpinn)

    for callback in callbacks:
        callback.on_train_begin()
    if mode == "layer":
        layer_train(
            fbpinn,
            epochs,
            start_epoch=start_epoch,
            callbacks=callbacks,
            val_truth=val_truth,
            patience=patience,
            log_interval=log_interval,
            eval_interval=eval_interval,
            val_input=val_input,
            png_salt=png_salt,
            layer_scheduler=layer_scheduler,
            loss_scheduler=loss_scheduler,
            path_to_ckpt=path_to_ckpt,
            lr_scheduler=lr_scheduler,
            configs=configs,
        )
    else:
        raise ValueError("Unsupported train mode")
    for callback in callbacks:
        callback.on_train_end()
