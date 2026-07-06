import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments.complex_geometry.cylinder_geometry import prepare_geometry, scale
from geofbpinn.geometry.polygon_decomposition import Decomposition2DPolygon
from experiments.complex_geometry.functions.cylinder_inviscid import CylinderInviscid
from experiments.complex_geometry.functions.cylinder_viscid import CylinderViscid
from geofbpinn.networks.topology.fbpinn.model import FBPINN


phys_func = CylinderViscid
model = "FBPINN_full_7867"
CHECKPOINT_DIR = "../checkpoints/" + model
OUTPUT_DIR = "../predictions"
POINT_STEP = 2
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch on cuda:", torch.cuda.is_available())

domain, hole = prepare_geometry()
block_size = (0.4, 0.4)
overlap = (0.15, 0.15)
bbox_left = (-block_size[0] / 4, -block_size[1] / 4)
bbox_right = (2000 * scale + block_size[0] / 4, 1200 * scale + block_size[1] / 4)
block_scales = {"vx": 0.003505, "vy": 0.00129, "pressure": 1.69e-05}
block_shifts = {
    "vx": 0.0072,
    "vy": -1.9e-06,
    "pressure": -7.9e-06,
}
output_scale = torch.tensor(
    [block_scales["pressure"], block_scales["vx"], block_scales["vy"]],
    device=device,
    requires_grad=False,
)
output_shift = torch.tensor(
    [block_shifts["pressure"], block_shifts["vx"], block_shifts["vy"]],
    device=device,
    requires_grad=False,
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
dec.remove_redundant_blocks(samples_per_block=2000, tol=0.0001, verbose=False)
print(f"Decomposition has {len(dec.blocks)} blocks")

model_config = {
    "input_size": 2,
    "output_size": 3,
    "activation_func": ["tanh", "tanh", "linear"],
    "models_size": [64, 64],
    "device": device,
    "weight": torch.nn.init.xavier_uniform_,
    "biases": torch.nn.init.zeros_,
}
pde = phys_func(cylinder=hole, scale=scale, device=device, path_to_data="../")
nn = FBPINN(
    **model_config,
    equation=pde,
    physic_loss=pde.phys_loss,
    boundary_loss=pde.sub_losses,
    decomposition=dec,
)
nn.to(device)

# path_to_ckpt = CHECKPOINT_DIR + "/fbpinn123_29000.weights.h5"
path_to_ckpt = CHECKPOINT_DIR + "/fbpinn123_best.weights.h5"
nn.load_weights(path_to_ckpt)
nn.eval()

x_np = pde.val_input[::POINT_STEP]
y = pde.solution(x_np)
x = torch.tensor(x_np, device=device)

with torch.no_grad():
    pred = nn(x)

pred_np = pred.cpu().numpy()
y_np = y.cpu().numpy()

range_mae = abs(pred_np - y_np)
for i in range(3):
    field_range = y_np[:, i].max() - y_np[:, i].min()
    if field_range > 0:
        range_mae[:, i] /= field_range

fields = ["pressure", "vx", "vy"]
field_labels = ["$p'$", "$v_x$", "$v_y$"]

for i, (field, label) in enumerate(zip(fields, field_labels)):
    vmin = min(pred_np[:, i].min(), y_np[:, i].min())
    vmax = max(pred_np[:, i].max(), y_np[:, i].max())
    loss_vmax = range_mae[:, i].max()
    loss_vmin = range_mae[:, i].min()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Field: {label}", fontsize=14)

    sc0 = axes[0].scatter(
        x_np[:, 0],
        x_np[:, 1],
        c=pred_np[:, i],
        cmap="viridis",
        s=10,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"Predicted {label}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, linewidth=0.3)
    fig.colorbar(sc0, ax=axes[0], shrink=0.8, pad=0.05)

    sc1 = axes[1].scatter(
        x_np[:, 0],
        x_np[:, 1],
        c=y_np[:, i],
        cmap="viridis",
        s=10,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"True {label}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].grid(True, linewidth=0.3)
    fig.colorbar(sc1, ax=axes[1], shrink=0.8, pad=0.05)

    sc2 = axes[2].scatter(
        x_np[:, 0],
        x_np[:, 1],
        c=range_mae[:, i],
        cmap="viridis",
        s=10,
        vmin=0,
        vmax=loss_vmax,
    )
    axes[2].set_title(f"Range MAE {label}")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].grid(True, linewidth=0.3)
    fig.colorbar(sc2, ax=axes[2], shrink=0.8, pad=0.05)

    plt.tight_layout()
    output_path = OUTPUT_DIR + f"/{model}_{field}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}, loss_max {loss_vmax}")

print("Losses")
diff = y_np - pred_np
abs_diff = np.abs(diff)  # (N, d)
abs_true = np.abs(y_np)  # (N, d)

val_mse_loss = np.mean(np.square(diff))
val_mae_loss = np.mean(abs_diff)
val_l1_loss = np.mean(abs_diff) / np.mean(abs_true)

per_component_mae = abs_diff.mean(axis=0)  # (d,)
true_range = y_np.max(axis=0) - y_np.min(axis=0)  # (d,)
per_component_range_mae = per_component_mae / true_range  # (d,)
val_range_mae = per_component_range_mae.mean()

res = {
    "Validation MSE loss": val_mse_loss,
    "Validation MAE loss": val_mae_loss,
    "Validation Relative L1Loss": val_l1_loss,
    "Validation Range MAE": val_range_mae,
}
for k, v in res.items():
    print(k, v)
