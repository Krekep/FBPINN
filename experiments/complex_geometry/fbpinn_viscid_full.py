import os
import numpy as np
import torch
import mlflow
import random

from experiments.complex_geometry.utils.flat_dict import flatten_dict
from geofbpinn.utils.get_classes import (
    get_layer_scheduler,
    get_loss_scheduler,
    get_lr_scheduler,
    get_embedding,
    get_decomposition,
)
from experiments.complex_geometry.cylinder_geometry import scale
from geofbpinn.geometry.base_decomposition import BaseDecomposition
from geofbpinn.geometry.plot import plot_decomposition2d
from geofbpinn.networks.topology.fbpinn.model import FBPINN
from functions.cylinder_viscid import CylinderViscid
from geofbpinn.networks.topology.fbpinn.trainer import train_fbpinn
from cfg.decomposition_config import domain, hole, decomposition_config
from cfg.env_config import env_config
from cfg.model_config import compile_config, embedding_config, model_config
from cfg.train_config import scheduler_config, train_config
from geofbpinn.utils.checkpoint import save_checkpoint

configs = {
    "decomposition": decomposition_config,
    "train": train_config,
    "model": model_config,
    "embedding": embedding_config,
    "env": env_config,
    "compile": compile_config,
    "scheduler": scheduler_config,
}

run_id = random.randint(1, 10000)
torch.manual_seed(env_config["random_seed"])
random.seed(env_config["random_seed"])
np.random.seed(env_config["random_seed"])

device = env_config["device"]
if device == "cuda":
    torch.cuda.manual_seed(env_config["random_seed"])
    torch.cuda.manual_seed_all(env_config["random_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

v_inf = 0.0075
dec: BaseDecomposition = get_decomposition(env_config["Decomposition"])(
    **decomposition_config,
    device=device,
)
dec.remove_redundant_blocks(samples_per_block=2000, tol=0.0001, verbose=False)
plot_decomposition2d(
    dec.blocks,
    polygon_vertices=domain,
    holes=[hole],
    figsize=(12, 4),
)
print(f"Decomposition has {len(dec.blocks)} blocks")
configs["decomposition"] = dec.get_config()

pde = CylinderViscid(cylinder=hole, scale=scale, v_inf=v_inf, device=device)
phys_loss = pde.phys_loss
which = pde.description

mlflow.set_experiment(env_config["experiment_name"])

run_name = f"{env_config['model_name']}_{run_id}"
mlflow.start_run(run_name=run_name, log_system_metrics=True)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("Class", pde.equation_class)
mlflow.log_param("v_inf", v_inf)
mlflow.log_params(flatten_dict(configs))
mlflow.log_param("decomposition_omega", dec.omega)
mlflow.log_param("decomposition_overlap", dec.overlap)
os.makedirs(f"./checkpoints/{run_name}", exist_ok=True)
logdir = f"./logs/{run_name}"

embedding = get_embedding(env_config["embedding"])(**embedding_config)
nn = FBPINN(
    **model_config,
    equation=pde,
    embedding=embedding,
    physic_loss=phys_loss,
    boundary_loss=pde.sub_losses,
    decomposition=dec,
    device=device,
)
nn.to(device)
nn.custom_compile(**compile_config)
mlflow.log_param("Number of submodels", len(nn.blocks))
mlflow.log_param("Num blocks per axis", nn.decomposition.num_blocks_per_axis)
mlflow.log_param("Blocks per axis", list(map(len, nn.decomposition.blocks_per_axis)))
b_per_a = list(map(len, nn.decomposition.blocks_per_axis))
print("Blocks per axis", b_per_a)

x = pde.val_input[::10]
y = torch.tensor(pde.solution(x), device=device)
x = torch.tensor(x, device=device)
y_pred_before_train = nn.call(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

layer_scheduler_config = {
    "n": len(nn.blocks),
    # "first": (0, sum(b_per_a) // 2 + b_per_a[-1]),
    # "second": (sum(b_per_a) // 2 - b_per_a[-1], sum(b_per_a)),
    # "step": 60000
}
layer_scheduler = get_layer_scheduler(env_config["layer_scheduler"])(
    **layer_scheduler_config
)
mlflow.log_params(layer_scheduler_config)

loss_scheduler_config = {
    "k": 0,
    "boundary_indices": list(range(len(pde.sub_losses))),
    "loss_weights": [10, 1, 1, 1, 1, 100],
}
loss_scheduler = get_loss_scheduler(env_config["loss_scheduler"])(
    **loss_scheduler_config
)
mlflow.log_params(loss_scheduler_config)

train_config["layer_scheduler"] = layer_scheduler
train_config["loss_scheduler"] = loss_scheduler
train_config["path_to_ckpt"] = f"./checkpoints/{run_name}"
scheduler = None
if env_config["scheduler"] is not None:
    scheduler = get_lr_scheduler(env_config["scheduler"])(
        nn.optimizer, **scheduler_config
    )
train_fbpinn(
    **train_config,
    fbpinn=nn,
    callbacks=[],
    val_truth=y,
    val_input=x,
    png_salt=str("123"),
    lr_scheduler=scheduler,
    configs=configs,
)
save_checkpoint(
    nn,
    f"./checkpoints/{run_name}/last.weights.h5",
    configs,
    nn.optimizer,
    scheduler,
    train_config["epochs"],
)

mlflow.end_run()
