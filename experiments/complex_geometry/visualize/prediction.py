import os
import re
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
os.environ["KERAS_BACKEND"] = "torch"

from experiments.complex_geometry.cylinder_geometry import prepare_geometry, scale
from src.geometry.polygon_decomposition import Decomposition2DPolygon
from experiments.complex_geometry.functions.cylinder_inviscid import CylinderInviscid
from experiments.complex_geometry.functions.cylinder_viscid import CylinderViscid
from src.networks.topology.fbpinn.model import FBPINN


def scatter_grid(axes_row, xy, values_list, titles, vranges):
    scs = []
    for ax, val, title, (vmin, vmax) in zip(axes_row, values_list, titles, vranges):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, cmap="viridis", s=10, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linewidth=0.3)
        scs.append(sc)
    return scs


phys_func = CylinderViscid
model = "FBPINN_full_5957"
CHECKPOINT_DIR = "../checkpoints/" + model
OUTPUT_DIR = "../predictions"
POINT_STEP = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch on cuda:", torch.cuda.is_available())

domain, hole = prepare_geometry()
block_size = (0.25, 0.25)
overlap = (0.08, 0.08)
bbox_left = (-block_size[0] / 2, -block_size[1] / 2)
bbox_right = (2000 * scale + block_size[0] / 2, 1200 * scale + block_size[1] / 2)
block_scales = {
  "vx": 0.003505,
  "vy": 0.00129,
  "pressure": 1.69e-05
}
block_shifts = {
  "vx":  0.0072,
  "vy": -1.9e-06,
  "pressure": -7.9e-06,
}
output_scale = torch.tensor(
    [block_scales["pressure"], block_scales["vx"], block_scales["vy"]],
    device=device, requires_grad=False
)
output_shift = torch.tensor(
    [block_shifts["pressure"], block_shifts["vx"], block_shifts["vy"]],
    device=device, requires_grad=False
)

dec = Decomposition2DPolygon(
    polygon_vertices=domain,
    bbox_left=bbox_left,
    bbox_right=bbox_right,
    block_scales=output_scale,
    block_shift=output_shift,
    block_size=block_size,
    overlap=overlap,
    points_per_block=1,
    eps_full=1e-3,
    holes=[hole],
    device=device,
)
print(f"Decomposition has {len(dec.blocks)} blocks")

model_config = {
    "input_size": 2,
    "output_size": 3,
    "activation_func": ["tanh", "tanh", "tanh", "linear"],
    "models_size": [24, 24, 24],
    "device": device,
    "weight": torch.nn.init.xavier_uniform_,
    "biases": torch.nn.init.zeros_,
}
pde = phys_func(cylinder=hole, scale=scale, device=device, path_to_data="../")
nn = FBPINN(
    **model_config,
    physic_loss=pde.phys_loss,
    boundary_loss=pde.sub_losses,
    decomposition=dec,
)
nn.to(device)

path_to_ckpt = CHECKPOINT_DIR + "/fbpinn123_2000.weights.h5"
# path_to_ckpt = CHECKPOINT_DIR + "/fbpinn123_best.weights.h5"
nn.load_weights(path_to_ckpt)
nn.eval()

x_np = pde.val_input[::POINT_STEP]
y = pde.solution(x_np)
x = torch.tensor(x_np, device=device)

with torch.no_grad():
    pred = nn(x)
rel_l1 = (
    torch.abs(y - pred)
    / torch.maximum(torch.abs(y), torch.tensor(1e-8))
).cpu().numpy()
pred_np = pred.cpu().numpy()
y_np = y.cpu().numpy()

fig, axes = plt.subplots(3, 3, figsize=(18, 14))

vmin_list = []
vmax_list = []
for i in range(3):
    vmin = min(pred_np[:, i].min(), y_np[:, i].min())
    vmax = max(pred_np[:, i].max(), y_np[:, i].max())
    vmin_list.append(vmin)
    vmax_list.append(vmax)

fields = ["pressure", "vx", "vy"]
scs_pred1 = axes[0, 0].scatter(x_np[:, 0], x_np[:, 1], c=pred_np[:, 0], cmap="viridis", s=10, vmin=vmin_list[0], vmax=vmax_list[0])
axes[0, 0].set_title(f"Predicted {fields[0]}")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
axes[0, 0].grid(True, linewidth=0.3)
scs_pred2 = axes[0, 1].scatter(x_np[:, 0], x_np[:, 1], c=pred_np[:, 1], cmap="viridis", s=10, vmin=vmin_list[1], vmax=vmax_list[1])
axes[0, 1].set_title(f"Predicted {fields[1]}")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
axes[0, 1].grid(True, linewidth=0.3)
scs_pred3 = axes[0, 2].scatter(x_np[:, 0], x_np[:, 1], c=pred_np[:, 2], cmap="viridis", s=10, vmin=vmin_list[2], vmax=vmax_list[2])
axes[0, 2].set_title(f"Predicted {fields[2]}")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")
axes[0, 2].grid(True, linewidth=0.3)
scs_true1 = axes[1, 0].scatter(x_np[:, 0], x_np[:, 1], c=y_np[:, 0], cmap="viridis", s=10, vmin=vmin_list[0], vmax=vmax_list[0])
axes[1, 0].set_title(f"True {fields[0]}")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
axes[1, 0].grid(True, linewidth=0.3)
scs_true2 = axes[1, 1].scatter(x_np[:, 0], x_np[:, 1], c=y_np[:, 1], cmap="viridis", s=10, vmin=vmin_list[1], vmax=vmax_list[1])
axes[1, 1].set_title(f"True {fields[1]}")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
axes[1, 1].grid(True, linewidth=0.3)
scs_true3 = axes[1, 2].scatter(x_np[:, 0], x_np[:, 1], c=y_np[:, 2], cmap="viridis", s=10, vmin=vmin_list[2], vmax=vmax_list[2])
axes[1, 2].set_title(f"True {fields[2]}")
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("y")
axes[1, 2].grid(True, linewidth=0.3)

scs_err = axes[2, 0].scatter(x_np[:, 0], x_np[:, 1], c=rel_l1[:, 0], cmap="viridis", s=10, vmin=0, vmax=2)
axes[2, 0].set_title(f"Error {fields[0]}")
axes[2, 0].set_xlabel("x")
axes[2, 0].set_ylabel("y")
axes[2, 0].grid(True, linewidth=0.3)
axes[2, 1].scatter(x_np[:, 0], x_np[:, 1], c=rel_l1[:, 1], cmap="viridis", s=10, vmin=0, vmax=2)
axes[2, 1].set_title(f"Error {fields[1]}")
axes[2, 1].set_xlabel("x")
axes[2, 1].set_ylabel("y")
axes[2, 1].grid(True, linewidth=0.3)
axes[2, 2].scatter(x_np[:, 0], x_np[:, 1], c=rel_l1[:, 2], cmap="viridis", s=10, vmin=0, vmax=2)
axes[2, 2].set_title(f"Error {fields[2]}")
axes[2, 2].set_xlabel("x")
axes[2, 2].set_ylabel("y")
axes[2, 2].grid(True, linewidth=0.3)

fig.colorbar(scs_pred1, ax=axes[0, 0], location="right", shrink=0.8, pad=0.05)
fig.colorbar(scs_pred2, ax=axes[0, 1], location="right", shrink=0.8, pad=0.05)
fig.colorbar(scs_pred3, ax=axes[0, 2], location="right", shrink=0.8, pad=0.05)
fig.colorbar(scs_true1, ax=axes[1, 0], location="right", shrink=0.8, pad=0.05)
fig.colorbar(scs_true2, ax=axes[1, 1], location="right", shrink=0.8, pad=0.05)
fig.colorbar(scs_true3, ax=axes[1, 2], location="right", shrink=0.8, pad=0.05)
fig.colorbar(scs_err, ax=axes[2, :], location="right", shrink=0.8, pad=0.05)

output_path = OUTPUT_DIR + f"/{model}.png"
plt.savefig(output_path, dpi=450)
plt.close()
