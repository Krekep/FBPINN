import glob
import os
import random

import numpy as np
import torch
import mlflow

from experiments.complex_geometry.utils.flat_dict import flatten_dict
from experiments.complex_geometry.cylinder_geometry import scale
from functions.cylinder_viscid import CylinderViscid
from geofbpinn.utils.get_classes import get_layer_scheduler, get_loss_scheduler
from geofbpinn.utils.checkpoint import load_checkpoint, save_checkpoint
from geofbpinn.networks.topology.fbpinn.trainer import train_fbpinn
from cfg.decomposition_config import hole
from cfg.env_config import env_config

CKPT_ROOT = "./checkpoints"
RUN_DIR = "./checkpoints/FBPINN_full_7889"
ADDITIONAL_EPOCHS = 20000
V_INF = 0.0075


def find_latest_checkpoint(root: str, run_dir: str = None) -> str:
    search_root = run_dir if run_dir else root
    pattern = os.path.join(search_root, "**", "*.weights.h5")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        raise FileNotFoundError(f"There is no checkpoints: {pattern}")
    return max(paths, key=os.path.getmtime)


device = env_config["device"]
torch.manual_seed(env_config["random_seed"])
random.seed(env_config["random_seed"])
np.random.seed(env_config["random_seed"])
if device == "cuda":
    torch.cuda.manual_seed_all(env_config["random_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ckpt_path = find_latest_checkpoint(CKPT_ROOT, RUN_DIR)
run_dir = os.path.dirname(ckpt_path)
print(f"Resume from: {ckpt_path}")

pde = CylinderViscid(cylinder=hole, scale=scale, v_inf=V_INF, device=device)

nn, lr_scheduler, checkpoint, start_epoch = load_checkpoint(
    ckpt_path, pde, device=device
)
configs = checkpoint["configs"]
train_config = configs["train"]

print(f"Loaded epoch={start_epoch}, blocks={len(nn.blocks)}")

mlflow.set_experiment(env_config["experiment_name"])
mlflow.start_run(
    run_name=f"{env_config['model_name']}_resume_{start_epoch}",
    log_system_metrics=True,
)
mlflow.set_tag("Training Info", f"Resumed from epoch {start_epoch}")
mlflow.log_param("resumed_from_checkpoint", ckpt_path)
mlflow.log_params(flatten_dict(configs))

layer_scheduler = get_layer_scheduler(env_config["layer_scheduler"])(n=len(nn.blocks))

loss_scheduler_config = {
    "k": 0,
    "boundary_indices": list(range(len(pde.sub_losses))),
    "loss_weights": [10, 1, 1, 1, 1, 100],
}
loss_scheduler = get_loss_scheduler(env_config["loss_scheduler"])(
    **loss_scheduler_config
)
mlflow.log_params(loss_scheduler_config)

x = pde.val_input[::10]
y = torch.tensor(pde.solution(x), device=device)
x = torch.tensor(x, device=device)

train_config["layer_scheduler"] = layer_scheduler
train_config["loss_scheduler"] = loss_scheduler
train_config["path_to_ckpt"] = run_dir
train_config["start_epoch"] = start_epoch + 1
train_config["epochs"] = start_epoch + 1 + ADDITIONAL_EPOCHS

train_fbpinn(
    **train_config,
    fbpinn=nn,
    callbacks=[],
    val_truth=y,
    val_input=x,
    png_salt="resume",
    lr_scheduler=lr_scheduler,
    configs=configs,
)

save_checkpoint(
    nn,
    f"{run_dir}/last_resumed.weights.h5",
    configs,
    nn.optimizer,
    lr_scheduler,
    train_config["epochs"],
)
mlflow.end_run()
