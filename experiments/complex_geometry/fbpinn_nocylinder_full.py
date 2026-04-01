import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ["KERAS_BACKEND"] = "torch"
import torch
# torch.cuda.is_available = lambda: False

import random
from matplotlib import pyplot as plt

from experiments.complex_geometry.cylinder_geometry import prepare_geometry, scale
from src.geometry.polygon_decomposition import Decomposition2DPolygon
from src.networks.schedulers.layer import BaseLayerScheduler
from src.networks.schedulers.loss import AdaptiveLossScheduler
from src.networks.topology.fbpinn.model import FBPINN
from functions.euler_empty import NoObstacleInviscid
from src.networks.topology.fbpinn.trainer import train_fbpinn
from src.networks.schedulers.lr import WarmupReduceLROnPlateau
import mlflow
print("Torch on cuda", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
domain, hole = prepare_geometry()
v_inf = 0.0075
block_size = (0.4, 0.4)
overlap = (0.15, 0.15)
bbox_left = (-block_size[0] / 2, -block_size[1] / 2)
bbox_right = (2000 * scale + block_size[0] / 2, 1200 * scale + block_size[1] / 2)
points_per_block = 200
eps_full = 1e-3
x_lim = 1.0
y_lim = 1.0
lc = [0, 0]
rc = [x_lim, y_lim]
lr = 1e-4
block_scales = {
  "vx": 0.00137,
  "vy": 0.00142,
  "pressure": 1.46e-5
}
block_shifts = {
  "vx":  0.007,
  "vy": -3.5e-6,
  "pressure": -2.93e-6,
}
output_scale = torch.tensor([block_scales["pressure"], block_scales["vx"], block_scales["vy"]], device=device, requires_grad=False)
output_shift = torch.tensor([block_shifts["pressure"], block_shifts["vx"], block_shifts["vy"]], device=device, requires_grad=False)

dec = Decomposition2DPolygon(
    polygon_vertices=domain,
    bbox_left=bbox_left,
    bbox_right=bbox_right,
    block_scales=output_scale,
    block_shift=output_shift,
    block_size=block_size,
    overlap=overlap,
    points_per_block=points_per_block,
    eps_full=eps_full,
    device=device
)
# dec.remove_redundant_blocks(samples_per_block=2000, tol=0.0001, verbose=False)
# plot_decomposition(dec.blocks, polygon_vertices=domain, holes=[hole], figsize=(12, 4), savepath="decomposition_airfoil.png")
print(f"Decomposition has {len(dec.blocks)} blocks")

model_config = {
    "input_size": 2,
    "output_size": 3,
    "activation_func": ["tanh", "tanh", "linear"],
    "models_size": [16, 16],
    "device": device,
    "weight": torch.nn.init.xavier_uniform_,
    "biases": torch.nn.init.zeros_
}

pde = NoObstacleInviscid(scale=scale, v_inf=v_inf, device=device)
phys_loss = pde.phys_loss
which = pde.description

mlflow.set_experiment("FBPINN Cyclinder")
run_id = random.randint(1, 10000)
run_name = f"FBPINN_full_{run_id}"
mlflow.start_run(run_name=run_name, log_system_metrics=True)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("Obstacle", "No cylinder")
mlflow.set_tag("Class", pde.equation_class)
mlflow.log_param("left_bound", bbox_left)
mlflow.log_param("right_bound", bbox_right)
mlflow.log_param("leaning_rate", lr)
mlflow.log_param("Block scales", block_scales)
mlflow.log_param("Block shifts", block_shifts)
mlflow.log_param("Block size", block_size)
mlflow.log_param("overlap", overlap)
mlflow.log_param("v_inf", v_inf)
mlflow.log_param("points_per_block", points_per_block)
mlflow.log_params(model_config)
os.makedirs(f"./checkpoints/{run_name}", exist_ok=True)
logdir = f"./logs/{run_name}"

nn = FBPINN(
    **model_config,
    physic_loss=phys_loss,
    boundary_loss=pde.sub_losses,
    decomposition=dec,
)
nn.to(device)
nn.custom_compile(
    optimizer="AdamW", rate=lr, loss_func="RelativeL1Loss", run_eagerly=False
)
mlflow.log_param("Number of submodels", len(nn.blocks))
mlflow.log_param("Num blocks per axis", nn.decomposition.num_blocks_per_axis)
mlflow.log_param("Blocks per axis", list(map(len, nn.decomposition.blocks_per_axis)))
print("Blocks per axis", list(map(len, nn.decomposition.blocks_per_axis)))

x = pde.val_input
y = torch.tensor(pde.solution(x), device=device)
x = torch.tensor(x, device=device)
y_pred_before_train = nn.call(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

layer_scheduler_config = {
    "n": len(nn.blocks),
}
layer_scheduler = BaseLayerScheduler(**layer_scheduler_config)
mlflow.log_param("Layer scheduler", "BaseLayerScheduler")
mlflow.log_params(layer_scheduler_config)

loss_scheduler_config = {
    "k": 0,
    "boundary_indices": list(range(len(pde.sub_losses))),
    "loss_weights": [1, 1, 1, 1, 1, 100],
    "threshold": 1e-3,
    "loss_multiplier": 10.0,
}
loss_scheduler = AdaptiveLossScheduler(**loss_scheduler_config)
mlflow.log_param("Loss scheduler", "AdaptiveLossScheduler")
mlflow.log_params(loss_scheduler_config)

train_config = {
    "epochs": 40000,
    "patience": 4_000_000,
    "eval_interval": 100,
    "batch_size": 10_000,
    "log_interval": 500,
    "mode": "layer",
    "layer_scheduler": layer_scheduler,
    "loss_scheduler": loss_scheduler,
    "path_to_ckpt": f"./checkpoints/{run_name}",
}
mlflow.log_params(train_config)
scheduler = WarmupReduceLROnPlateau(
    nn.optimizer,
    mode='min',
    factor=0.8,
    patience=100000,
    warmup_epochs=10,
    warmup_start_factor=0.1
)
train_fbpinn(
    **train_config,
    fbpinn=nn,
    callbacks=[],
    verbose=0,
    val_truth=y,
    val_input=x,
    png_salt=str("123"),
    lr_scheduler=scheduler
)
mlflow.end_run()
nn.save_weights(f"./checkpoints/{run_name}/cyclinder_1_layer.weights.h5")

x_plot = x[:, :]

fig, axes = plt.subplots(nrows=2, ncols=3)
predicted = nn(x_plot).detach().cpu()
x_plot = x_plot.detach().cpu()
y = y[:, :].detach().cpu()
axes[0, 0].scatter(
    x_plot[:, 0:1], x_plot[:, 1:2], c=predicted[:, 0], cmap='viridis'
)
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
axes[0, 0].grid()
axes[0, 0].set_title("Predicted pressure")

axes[0, 1].scatter(
    x_plot[:, 0:1], x_plot[:, 1:2], c=predicted[:, 1], cmap='viridis'
)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
axes[0, 1].grid()
axes[0, 1].set_title("Predicted vx")

axes[0, 2].scatter(
    x_plot[:, 0:1], x_plot[:, 1:2], c=predicted[:, 2], cmap='viridis'
)
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")
axes[0, 2].grid()
axes[0, 2].set_title("Predicted vy")

axes[1, 0].scatter(
    x_plot[:, 0:1], x_plot[:, 1:2], c=y[:, 0], cmap='viridis'
)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
axes[1, 0].grid()
axes[1, 0].set_title("Truth pressure")

axes[1, 1].scatter(
    x_plot[:, 0:1], x_plot[:, 1:2], c=y[:, 1], cmap='viridis'
)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
axes[1, 1].grid()
axes[1, 1].set_title("Truth vx")

axes[1, 2].scatter(
    x_plot[:, 0:1], x_plot[:, 1:2], c=y[:, 2], cmap='viridis'
)
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("y")
axes[1, 2].grid()
axes[1, 2].set_title("Truth vy")
plt.savefig(f"Solution{run_name}", dpi=400)
plt.savefig(f"Solution{run_name}.eps", dpi=400, format="eps")
plt.show()

