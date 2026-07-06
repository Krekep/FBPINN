import os
import re
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments.complex_geometry.functions.poisson_3d import Poisson3D
from geofbpinn.geometry import DecompositionND, RectangleDomain
from experiments.complex_geometry.cylinder_geometry import prepare_geometry, scale
from geofbpinn.geometry.polygon_decomposition import Decomposition2DPolygon
from experiments.complex_geometry.functions.cylinder_inviscid import CylinderInviscid
from experiments.complex_geometry.functions.cylinder_viscid import CylinderViscid
from geofbpinn.networks.topology.fbpinn.model import FBPINN


def scatter_grid(axes_row, xy, values_list, titles, vranges):
    scs = []
    for ax, val, title, (vmin, vmax) in zip(axes_row, values_list, titles, vranges):
        sc = ax.scatter(
            xy[:, 0], xy[:, 1], c=val, cmap="viridis", s=10, vmin=vmin, vmax=vmax
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linewidth=0.3)
        scs.append(sc)
    return scs


phys_func = Poisson3D
model = "FBPINN_full_533"
CHECKPOINT_DIR = "../checkpoints/" + model
OUTPUT_DIR = "../predictions"
POINT_STEP = 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch on cuda:", torch.cuda.is_available())

block_size = (0.5, 0.5, 0.5)
overlap = (0.15, 0.15, 0.15)
bbox_left = (-block_size[0] / 2, -block_size[1] / 2, -block_size[2] / 2)
bbox_right = (1 + block_size[0] / 2, 1 + block_size[1] / 2, 1 + block_size[2] / 2)
block_scales = {"u": 1}
block_shifts = {"u": 0}
output_scale = torch.tensor([block_scales["u"]], device=device, requires_grad=False)
output_shift = torch.tensor([block_shifts["u"]], device=device, requires_grad=False)
domain = RectangleDomain([0, 0, 0], [1, 1, 1])
dec = DecompositionND(
    domain=domain,
    bbox_left=bbox_left,
    bbox_right=bbox_right,
    block_scales=output_scale,
    block_shift=output_shift,
    block_size=block_size,
    overlap=overlap,
    device=device,
)
print(f"Decomposition has {len(dec.blocks)} blocks")

model_config = {
    "input_size": 3,
    "output_size": 1,
    "activation_func": ["tanh", "tanh", "linear"],
    "models_size": [32, 32],
    "device": device,
    "weight": torch.nn.init.xavier_uniform_,
    "biases": torch.nn.init.zeros_,
}
pde = phys_func(device=device)
nn = FBPINN(
    **model_config,
    physic_loss=pde.phys_loss,
    boundary_loss=pde.sub_losses,
    decomposition=dec,
)
nn.to(device)

# path_to_ckpt = CHECKPOINT_DIR + "/fbpinn123_2000.weights.h5"
path_to_ckpt = CHECKPOINT_DIR + "/fbpinn123_best.weights.h5"
nn.load_weights(path_to_ckpt)
nn.eval()

x_np = pde.val_input[::POINT_STEP]
y = pde.solution(x_np)
x = torch.tensor(x_np, device=device)

with torch.no_grad():
    pred = nn(x)
rel_l1 = (
    (torch.abs(y - pred) / torch.maximum(torch.abs(y), torch.tensor(1e-8)))
    .cpu()
    .numpy()
)
pred_np = pred.cpu().numpy()
y_np = y.cpu().numpy()

vmin = min(pred_np[:, 0].min(), y_np[:, 0].min())
vmax = max(pred_np[:, 0].max(), y_np[:, 0].max())

z_slices = [0.0, 0.5063291, 1.0]
fig, axes = plt.subplots(3, 3, figsize=(14, 12))

for col, z_val in enumerate(z_slices):
    mask = np.isclose(x_np[:, 2], z_val)
    n = int(round(np.sqrt(mask.sum())))
    p = pred_np[mask, 0]
    t = y_np[mask, 0]
    e = rel_l1[mask, 0]

    sc_p = axes[0, col].scatter(
        x_np[mask, 0], x_np[mask, 1], c=p, vmin=vmin, vmax=vmax, cmap="viridis"
    )
    axes[0, col].set_title(f"Predicted u, z={z_val}")
    sc_t = axes[1, col].scatter(
        x_np[mask, 0], x_np[mask, 1], c=t, vmin=vmin, vmax=vmax, cmap="viridis"
    )
    axes[1, col].set_title(f"True u, z={z_val}")
    sc_e = axes[2, col].scatter(
        x_np[mask, 0], x_np[mask, 1], c=e, vmin=0, vmax=2, cmap="viridis"
    )
    axes[2, col].set_title(f"Error u, z={z_val}")

fig.colorbar(sc_p, ax=axes[0, :], location="right", shrink=0.8, pad=0.05)
fig.colorbar(sc_t, ax=axes[1, :], location="right", shrink=0.8, pad=0.05)
fig.colorbar(sc_e, ax=axes[2, :], location="right", shrink=0.8, pad=0.05)

output_path = OUTPUT_DIR + f"/{model}.png"
plt.savefig(output_path, dpi=450)
plt.close()
