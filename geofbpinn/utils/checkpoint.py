import torch

from geofbpinn.networks.topology.fbpinn.model import FBPINN
from geofbpinn.utils.get_classes import (
    get_decomposition,
    get_embedding,
    get_lr_scheduler,
)


def save_checkpoint(
    model: FBPINN, path: str, configs: dict, optimizer, scheduler, epoch: int
):
    model._scatter_weights()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "configs": configs,
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else None,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, pde, device="cpu"):
    """
    This function reconstructs the FBPINN model, its optimizer, and the learning
    rate scheduler exactly as they were at the time of saving. It also restores
    the epoch number and all configuration dictionaries.

    Parameters
    ----------
    path : str
        Path to the checkpoint file (.pt or .pth).
    pde : object
        PhysLoss object
    device : str, optional
        Device to load the model onto ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns
    -------
    tuple
        A tuple containing:
            - nn (FBPINN): The loaded FBPINN model with its internal optimizer
              already restored
            - scheduler (object or None): The restored learning rate scheduler
              if it exists in the checkpoint; otherwise, None.
            - checkpoint (dict): The raw loaded checkpoint dictionary, which
              includes the 'configs' and any other metadata saved.
            - epoch (int): The epoch number at which the checkpoint was saved.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    epoch = checkpoint["epoch"]
    configs = checkpoint["configs"]

    decomp_cfg = configs["decomposition"]
    env_cfg = configs["env"]
    model_cfg = configs["model"]
    embed_cfg = configs["embedding"]
    compile_cfg = configs["compile"]
    train_cfg = configs["train"]

    scheduler_cfg = configs.get("scheduler", {})  # может отсутствовать

    dec = get_decomposition(env_cfg["Decomposition"])(**decomp_cfg, device=device)
    dec.remove_redundant_blocks(samples_per_block=2000, tol=0.0001, verbose=False)

    embedding = get_embedding(env_cfg["embedding"])(**embed_cfg)
    nn = FBPINN(
        **model_cfg,
        equation=pde,
        embedding=embedding,
        physic_loss=pde.phys_loss,
        boundary_loss=pde.sub_losses,
        decomposition=dec,
        device=device
    )
    nn.to(device)
    nn.custom_compile(**compile_cfg)
    nn.load_state_dict(checkpoint["model_state_dict"])
    nn._sync_stacked_params()
    if "optimizer_state_dict" in checkpoint:
        nn.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = None
    if env_cfg.get("scheduler") is not None:
        scheduler_class = get_lr_scheduler(env_cfg["scheduler"])
        scheduler = scheduler_class(nn.optimizer, **scheduler_cfg)
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return nn, scheduler, checkpoint, epoch
