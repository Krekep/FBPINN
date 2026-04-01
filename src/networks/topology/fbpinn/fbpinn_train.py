"""
Module that provide different strategies of FBPINN training.
Detailed description at `DOI: 10.3103/S0027134925702820`
"""
import datetime
import time
import typing

import mlflow

from src.networks.schedulers.layer import BaseLayerScheduler
from src.networks.schedulers.loss import LossScheduler
import torch

if typing.TYPE_CHECKING:
    from src.networks.topology.fbpinn.model import FBPINN


def layer_train(
        fbpinn: "FBPINN",
        epochs: int,
        val_input: torch.Tensor,
        val_truth: torch.Tensor,
        start_epoch: int = 0,
        patience: int = 300,
        callbacks=None,
        log_interval: int = 100,
        eval_interval: int = 1,
        png_salt: str = "",
        layer_scheduler: BaseLayerScheduler = None,
        loss_scheduler: LossScheduler = None,
        path_to_ckpt: str = "",
        lr_scheduler=None
) -> None:
    """
    Dependent simultaneous training.

    Parameters
    ----------
    fbpinn: FBPINN
        model for training
    epochs: int
        Number of epochs for training
    val_input: torch.Tensor
        Validation input
    val_truth: torch.Tensor
        Validation expected output
    start_epoch: int
        Training starts from this epoch
    patience: int
        Number of epochs before stopping training due to lack of improvement train loss
    callbacks: list
    log_interval: int
        Number of epoch between log results to console
    eval_interval: int
        Number of epoch between evaluation model on validation data
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
    """
    curr_patience = 0
    best_loss = 1e6
    best_val_loss = 1e6

    max_w_update = 1e6
    for epoch in range(start_epoch, epochs):
        for callback in callbacks:
            callback.on_epoch_begin(epoch)
        start_time = time.perf_counter()
        layer_scheduler.step()
        loss_scheduler.step()
        layer_indices = layer_scheduler.on_epoch_start()
        frozen_indices = layer_scheduler.get_frozen_indices()
        trainable_set = set(layer_indices) - set(frozen_indices)

        for i, (nn, _) in enumerate(fbpinn.blocks):
            requires = (i in trainable_set)
            for p in nn.parameters():
                p.requires_grad_(requires)

        loss_indices, loss_weights = loss_scheduler.on_epoch_start(
            curr_max_w_update=max_w_update
        )
        data_parts = [
            fbpinn.blocks[i][1].get_data()
            for i in layer_indices
            if i not in frozen_indices
        ]
        data = torch.cat(data_parts, dim=0).to(fbpinn.device)
        data.requires_grad_(True)

        assert data.numel() > 0, f"""Data for train is empty. 
        Train indices {layer_indices}, frozen {frozen_indices}"""
        loss = 0
        step_loss, losses, max_w_update = train_step(
            fbpinn, data, layer_indices, frozen_indices, loss_indices, loss_weights
        )
        loss += step_loss

        end_time = time.perf_counter()
        if epoch % eval_interval == 0:
            with torch.no_grad():
                val_metrics = get_val_score(fbpinn, val_input, val_truth)
                mlflow.log_metric(
                    f"Max layer weights gradient", max_w_update, step=epoch
                )
                log_validation_metrics(
                    fbpinn,
                    val_metrics,
                    epoch,
                    loss,
                    end_time - start_time,
                )
                if lr_scheduler is not None:
                    lr_scheduler.step(val_metrics["Validation MSE loss"])
                    # lr_scheduler.step()
                if val_metrics["Validation MSE loss"] < best_val_loss:
                    best_val_loss = val_metrics["Validation MSE loss"]
                    fbpinn.save_weights(f"{path_to_ckpt}/fbpinn{png_salt}_best.weights.h5")
        if epoch % log_interval == 0 or epoch == epochs - 1:
            with torch.no_grad():
                if epoch % eval_interval != 0:
                    val_metrics = get_val_score(fbpinn, val_input, val_truth)
                log_graphics(fbpinn, [None], val_input, val_truth, epoch, loss, png_salt, val_metrics)
                print(f"Epoch {epoch}, layer indices {layer_indices}")
                print(f"Epoch {epoch}, loss indices {loss_indices}")
                print(f"Epoch {epoch}, data length {len(data)}")
                print("Losses", losses)
                fbpinn.save_weights(f"{path_to_ckpt}/fbpinn{png_salt}_{epoch}.weights.h5")
        if loss < best_loss:
            best_loss = loss
            curr_patience = 0
        curr_patience += 1
        if curr_patience > patience:
            print(
                f"Too long, Current patience is {curr_patience}, all patience is {patience}, best loss {best_loss}"
            )
            break
    log_graphics(fbpinn, [None], val_input, val_truth, epoch, loss, png_salt)


def train_step(
        fbpinn: "FBPINN",
        data: torch.Tensor,
        layer_indices: list[int],
        frozen_indices: list[int],
        loss_indices: list[int],
        loss_weights: list[float]
) -> tuple[float, list[float], float]:
    """
    Custom train step

    Parameters
    ----------
    fbpinn: FBPINN
        model instance
    data: torch.Tensor
        points
    layer_indices: list[int]
         list of active models
    frozen_indices: list[int]
        list of active, but frozen models
    loss_indices: list[int]
        list of active losses
    loss_weights: list[float]
        list of weights for each loss function

    Returns
    -------
    loss: float
        accumulated loss
    losses: list[float]
        loss values per loss function
    max_grad: float
        maximum update gradient
    """
    trainable_set = set(layer_indices) - set(frozen_indices)
    fbpinn.zero_grad()
    for sw in fbpinn.stacked_w + fbpinn.stacked_b:
        sw.grad = None

    loss = torch.zeros(1, device=data.device)
    losses = []
    for i, (loss_func, models) in enumerate(fbpinn.model_per_loss):
        if i not in loss_indices:
            continue
        active_models = [fbpinn.blocks[m] for m in models if m in trainable_set]
        if not active_models:
            continue
        temp = loss_func(fbpinn, data, active_models=active_models) * loss_weights[i]
        losses.append(temp.item())
        loss += temp
    loss.backward()
    fbpinn._scatter_gradients()
    torch.nn.utils.clip_grad_norm_(fbpinn.parameters(), max_norm=1)
    fbpinn.optimizer.step()
    fbpinn._sync_stacked_params()
    max_grad = torch.stack([
        sw.grad.abs().max()
        for sw in fbpinn.stacked_w + fbpinn.stacked_b
        if sw.grad is not None
    ]).max().item()
    return (
        loss.item(),
        losses,
        max_grad,
    )


def get_val_score(fbpinn: "FBPINN", val_input: torch.Tensor, y_true: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Compute MSE, MAE and Relative L1 Loss

    Parameters
    ----------
    fbpinn: FBPINN
        model for evaluate
    val_input: torch.Tensor
        input
    y_true: torch.Tensor
        expected output

    Returns
    -------
    val_losses: dict[str, torch.Tensor]
        pairs of loss name and loss value
    """
    y_pred = fbpinn.call(val_input)
    diff = y_true - y_pred
    abs_diff = torch.abs(diff)  # (N, d)
    abs_true = torch.abs(y_true)  # (N, d)
    val_mse_loss = torch.mean(torch.square(diff))
    val_mae_loss = torch.mean(abs_diff)
    val_l1_loss = torch.mean(abs_diff) / torch.clamp(torch.mean(abs_true), min=1e-9)
    per_component_mae = abs_diff.mean(dim=0)  # (d,)
    per_component_base = abs_true.mean(dim=0)  # (d,)
    per_component_l1 = per_component_mae / torch.clamp(per_component_base, min=1e-9)
    res = {
        "Validation MSE loss": val_mse_loss,
        "Validation MAE loss": val_mae_loss,
        "Validation Relative L1Loss": val_l1_loss,
    }
    for i in range(per_component_l1.shape[-1]):
        res[f"val_rell1_{i}"] = per_component_l1[i]

    return res


def log_validation_metrics(fbpinn: "FBPINN", val_metrics: dict[str, torch.Tensor], epoch: int, loss: torch.Tensor, epoch_time: int):
    """
    Log metrics to mlflow
    """
    mlflow.log_metrics(val_metrics, step=epoch)
    mlflow.log_metric("Epoch time", epoch_time, step=epoch)
    mlflow.log_metric("Loss", loss, step=epoch)
    mlflow.log_metric("Learning rate", fbpinn.optimizer.param_groups[0]['lr'], step=epoch)


def log_graphics(fbpinn: "FBPINN", t, val_input, val_truth, epoch, loss, png_salt, val_metrics=None):
    """
    TODO: Method for drawn current model results
    """
    if val_metrics is None:
        val_metrics = get_val_score(fbpinn, val_input, val_truth)

    print(
        f"Epoch {epoch}, loss {loss}, {datetime.datetime.now()}. {val_metrics}"
    )
