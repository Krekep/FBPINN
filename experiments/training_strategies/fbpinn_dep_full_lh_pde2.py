import random
import os

os.environ["KERAS_BACKEND"] = "torch"
import torch
from matplotlib import pyplot as plt

import mlflow

from geofbpinn.networks.topology.fbpinn.trainer import train_fbpinn
from geofbpinn.geometry import RectangleDomain, DecompositionND
from geofbpinn.networks.topology.fbpinn.model import FBPINN
from geofbpinn.networks.schedulers.loss import LossScheduler, AdaptiveLossScheduler
from geofbpinn.networks.schedulers.layer import (
    SequenceBlockScheduler,
    BaseLayerScheduler,
)
from experiments.training_strategies.Functions.LH_PDE1 import LH_PDE1

device = "cuda" if torch.cuda.is_available() else "cpu"
x_lim = 1.0
t_lim = 1.0
domain_lc = [0, 0]
domain_rc = [t_lim, x_lim]
pde = LH_PDE1(end_time=t_lim, device=device)
phys_loss = pde.phys_loss
which = pde.description

mlflow.set_experiment("FBPINN PDE Wave 1D")
run_id = random.randint(1, 10000)
run_name = f"FBPINN_full_{run_id}"
mlflow.start_run(run_name=run_name)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("mlflow.runName", f"model{run_id}")
mlflow.set_tag("Class", pde.equation_class)
mlflow.log_param("Domain left corner", domain_lc)
mlflow.log_param("Domain right corner", domain_rc)

initial_rate = 1e-4
block_size = [0.6, 0.9]
overlap = [0.15, 0.25]
boundary_lc = [domain_lc[0] - block_size[0] / 2, domain_lc[1] - block_size[1] / 2]
boundary_rc = [domain_rc[0] + block_size[0] / 2, domain_rc[1] + block_size[1] / 2]
domain = RectangleDomain(left_corner=domain_lc, right_corner=domain_rc)
points_per_block = 1000
block_scales = {"y": 1}
block_shifts = {"y": 0}
output_scale = torch.tensor([block_scales["y"]], device=device, requires_grad=False)
output_shift = torch.tensor([block_shifts["y"]], device=device, requires_grad=False)


dec = DecompositionND(
    domain=domain,
    bbox_left=boundary_lc,
    bbox_right=boundary_rc,
    overlap=overlap,
    block_size=block_size,
    block_scales=output_scale,
    block_shift=output_shift,
    points_per_block=points_per_block,
    device=device,
)

model_config = {
    "input_size": 2,
    "output_size": 1,
    "activation_func": ["tanh", "tanh", "linear"],
    "models_size": [32, 32],
    "device": device,
    "weight": torch.nn.init.xavier_uniform_,
    "biases": torch.nn.init.zeros_,
}
mlflow.log_param("left_bound", boundary_lc)
mlflow.log_param("right_bound", boundary_rc)
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
    decomposition=dec,
    equation=pde,
    physic_loss=phys_loss,
    boundary_loss=pde.sub_losses,
)
print("Number of submodels", len(nn.blocks))
print("Blocks per axis", nn.decomposition.num_blocks_per_axis)
print("Blocks per layer", sum(nn.decomposition.num_blocks_per_axis[1:]))

nn.custom_compile(
    optimizer="AdamW", rate=initial_rate, loss_func="RelativeL1Loss", run_eagerly=False
)

x = pde.val_input
y = pde.solution(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

layer_scheduler_config = {
    "n": len(nn.blocks),
}
layer_scheduler = BaseLayerScheduler(**layer_scheduler_config)
mlflow.log_param("Layer scheduler", "BaseLayerScheduler")
mlflow.log_params(layer_scheduler_config)
loss_scheduler_config = {
    "k": 5000,
    "boundary_indices": list(range(len(pde.sub_losses))),
    "loss_weights": [10000, 10, 10, 10, 1],
}
loss_scheduler = LossScheduler(**loss_scheduler_config)
mlflow.log_param("Loss scheduler", "LossScheduler")
mlflow.log_params(loss_scheduler_config)

train_config = {
    "epochs": 180_000,
    "patience": 500_000,
    "eval_interval": 5,
    "log_interval": 1000,
    "mode": "layer",
    "layer_scheduler": layer_scheduler,
    "loss_scheduler": loss_scheduler,
    "start_epoch": 0,
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
    lr_scheduler=None,
)
loss_after_train = nn.evaluate(x, y, verbose=0)
print("Before", loss_before_train)
print("After", loss_after_train)
mlflow.log_metric("Loss before training", loss_before_train)
mlflow.log_metric("Loss after training", loss_after_train)
mlflow.end_run()
nn.save_weights(f"lh_pde_1_dep_full_{run_id}.weights.h5")

x_plot = torch.linspace(domain_lc[1], domain_rc[1], steps=10000, device=device)
for t_py in [i / 10 for i in range(0, int(10 * t_lim + 1))]:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
    t = torch.ones_like(x_plot, device=device) * t_py
    x = torch.stack([t, x_plot], dim=1)
    y = pde.solution(x)

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

    y_pred = nn(x_plot.unsqueeze(1)).detach().cpu().numpy()
    axes[1].plot(x, y.cpu(), label="Real", color="orange")
    axes[1].plot(x, y_pred, label="Predicted", color="green")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()
    axes[1].grid()
    plt.savefig(f"FBPINN_{str(run_id)}_t{str(t_py)}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)
