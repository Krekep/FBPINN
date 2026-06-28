import torch

from geofbpinn.networks.topology.fbpinn.model import FBPINN
from geofbpinn.utils.get_classes import (
    get_decomposition,
    get_embedding,
    get_lr_scheduler,
)


def save_checkpoint(model, path, configs, optimizer, scheduler, epoch):
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


def load_checkpoint(path, pde, device="cpu"):
    """
    Полностью восстанавливает состояние эксперимента из чекпоинта.

    Параметры
    ----------
    path : str
        Путь к файлу чекпоинта.
    device : str
        Устройство для загрузки ('cpu' или 'cuda').

    Возвращает
    ----------
    dict
        Содержит:
            - model: загруженная модель FBPINN с оптимизатором
            - optimizer: оптимизатор (доступен через model.optimizer)
            - scheduler: восстановленный планировщик (или None)
            - epoch: номер эпохи (если есть)
            - configs: словарь всех конфигов
            - extra: дополнительные данные (метрики и т.п.)
    """
    checkpoint = torch.load(path, map_location=device)
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
    )
    nn.to(device)
    nn.custom_compile(**compile_cfg)
    nn.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        nn.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = None
    if env_cfg.get("scheduler") is not None:
        scheduler_class = get_lr_scheduler(env_cfg["scheduler"])
        scheduler = scheduler_class(nn.optimizer, **scheduler_cfg)
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return nn, scheduler, checkpoint, epoch
