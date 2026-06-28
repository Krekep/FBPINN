import math
import os
import torch
import random

from matplotlib import pyplot as plt

from experiments.complex_geometry.cylinder_geometry import prepare_geometry, scale
from geofbpinn.geometry import DecompositionND, RectangleDomain
from geofbpinn.geometry.plot import plot_decomposition2d
from geofbpinn.networks.schedulers.layer import (
    BaseLayerScheduler,
)
from geofbpinn.networks.schedulers.loss import AdaptiveLossScheduler
from geofbpinn.networks.topology.fbpinn.model import FBPINN
from experiments.training_strategies.Functions.LF_ODE1 import LF_ODE_1
from geofbpinn.networks.topology.fbpinn.trainer import train_fbpinn
from geofbpinn.networks.schedulers.lr import WarmupReduceLROnPlateau
import mlflow

print("Torch on cuda", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-4
epochs = 100_000
x_lim = math.pi * 2
domain_lc = [0]
domain_rc = [x_lim]
block_size = [0.72]
overlap = [0.3]
boundary_lc = [domain_lc[0]]
boundary_rc = [domain_rc[0]]
domain = RectangleDomain(left_corner=domain_lc, right_corner=domain_rc)
points_per_block = 500
block_scales = {"y": 1}
block_shifts = {"y": 0}
output_scale = torch.tensor([block_scales["y"]], device=device, requires_grad=False)
output_shift = torch.tensor([block_shifts["y"]], device=device, requires_grad=False)

dec = DecompositionND(
    domain=domain,
    bbox_left=boundary_lc,
    bbox_right=boundary_rc,
    block_scales=output_scale,
    block_shift=output_shift,
    block_size=block_size,
    overlap=overlap,
    points_per_block=points_per_block,
    device=device,
)
print(f"Decomposition has {len(dec.blocks)} blocks")

model_config = {
    "input_size": 1,
    "output_size": 1,
    "activation_func": ["tanh", "tanh", "linear"],
    "models_size": [16, 16],
    "device": device,
    "weight": torch.nn.init.xavier_uniform_,
    "biases": torch.nn.init.zeros_,
}

ode = LF_ODE_1(device=device)
phys_loss = ode.phys_loss
which = ode.description

mlflow.set_experiment("FBPINN LF_ODE_1")
run_id = random.randint(1, 10000)
run_name = f"FBPINN_full_{run_id}"
mlflow.start_run(run_name=run_name, log_system_metrics=True)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.log_param("left_bound", boundary_lc)
mlflow.log_param("right_bound", boundary_rc)
mlflow.log_param("leaning_rate", lr)
mlflow.log_param("Block scales", block_scales)
mlflow.log_param("Block shifts", block_shifts)
mlflow.log_param("Block size", block_size)
mlflow.log_param("overlap", overlap)
mlflow.log_param("points_per_block", points_per_block)
mlflow.log_params(model_config)
os.makedirs(f"./checkpoints/{run_name}", exist_ok=True)
logdir = f"./logs/{run_name}"

nn = FBPINN(
    **model_config,
    equation=ode,
    physic_loss=phys_loss,
    boundary_loss=ode.sub_losses,
    decomposition=dec,
)
nn.to(device)
nn.custom_compile(
    optimizer="AdamW", rate=lr, loss_func="RelativeL1Loss", run_eagerly=False
)
mlflow.log_param("Number of submodels", len(nn.blocks))
mlflow.log_param("Num blocks per axis", nn.decomposition.num_blocks_per_axis)
mlflow.log_param("Blocks per axis", list(map(len, nn.decomposition.blocks_per_axis)))
b_per_a = list(map(len, nn.decomposition.blocks_per_axis))
print("Blocks per axis", b_per_a)

x = ode.val_input[::]
y = torch.tensor(ode.solution(x), device=device)
x = torch.tensor(x, device=device)
y_pred_before_train = nn.call(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

layer_scheduler_config = {"n": len(nn.blocks)}
layer_scheduler = BaseLayerScheduler(**layer_scheduler_config)
mlflow.log_param("Layer scheduler", "BaseLayerScheduler")
mlflow.log_params(layer_scheduler_config)

loss_scheduler_config = {
    "k": 300,
    "boundary_indices": list(range(len(ode.sub_losses))),
    "loss_weights": [1, 1],
    "threshold": 1e-3,
    "loss_multiplier": 10.0,
}
loss_scheduler = AdaptiveLossScheduler(**loss_scheduler_config)
mlflow.log_param("Loss scheduler", "AdaptiveLossScheduler")
mlflow.log_params(loss_scheduler_config)

train_config = {
    "epochs": epochs,
    "start_epoch": 0,
    "patience": 4_000_000,
    "eval_interval": 100,
    "log_interval": 500,
    "mode": "layer",
    "layer_scheduler": layer_scheduler,
    "loss_scheduler": loss_scheduler,
    "path_to_ckpt": f"./checkpoints/{run_name}",
}
mlflow.log_params(train_config)
train_fbpinn(
    **train_config,
    fbpinn=nn,
    callbacks=[],
    val_truth=y,
    val_input=x,
    png_salt=str("123"),
)
loss_after_train = nn.evaluate(x, y, verbose=0)
print("Before", loss_before_train)
print("After", loss_after_train)
mlflow.log_metric("Loss before training", loss_before_train)
mlflow.log_metric("Loss after training", loss_after_train)
mlflow.end_run()
nn.save_weights(f"./checkpoints/{run_name}/cyclinder_1_layer.weights.h5")


x_plot = x
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
y = ode.solution(x)

x_exp = x.unsqueeze(0)
vmins = nn.all_vmins.unsqueeze(1)
vmaxs = nn.all_vmaxs.unsqueeze(1)
x_norm = 2.0 * (x_exp - vmins) / (vmaxs - vmins) - 1.0

with torch.no_grad():
    outputs = nn._manual_forward(nn.stacked_w, nn.stacked_b, x_norm)

    scales = nn.all_scales.unsqueeze(1)
    shifts = nn.all_shifts.unsqueeze(1)
    outputs_phys = outputs * scales + shifts

    windows = nn.decomposition.batched_window(x)
    windowed = outputs_phys * windows

x = x_plot.detach().cpu().numpy()
axes[0].plot(x, y.cpu(), label="Real", color="orange")
for i in range(len(nn.blocks)):
    axes[0].plot(x, windowed[i].detach().cpu().numpy())
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()
axes[0].grid()
axes[0].set_title("Actual vs Predicted Data")

y_pred = nn(x_plot).detach().cpu().numpy()
axes[1].plot(x, y.cpu(), label="Real", color="orange")
axes[1].plot(x, y_pred, label="Predicted", color="green")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].legend()
axes[1].grid()
plt.savefig(f"FBPINN_{str(run_id)}.png", dpi=450, bbox_inches="tight")
plt.close(fig)
