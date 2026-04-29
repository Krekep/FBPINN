import os
import re
import random
import glob

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import pyplot as plt

os.environ["KERAS_BACKEND"] = "torch"

from experiments.complex_geometry.cylinder_geometry import prepare_geometry, scale
from geofbpinn.geometry.polygon_decomposition import Decomposition2DPolygon
from experiments.complex_geometry.functions.euler_empty import NoObstacleInviscid
from experiments.complex_geometry.functions.cylinder_inviscid import CylinderInviscid
from experiments.complex_geometry.functions.cylinder_viscid import CylinderViscid
from geofbpinn.networks.topology.fbpinn.model import FBPINN
from geofbpinn.geometry.plot import plot_decomposition2d


def scatter_grid(axes_row, xy, values_list, titles, vranges=None):
    """Fill a row of axes with scatter plots. Returns the scatter objects."""
    scs = []
    if vranges is None:
        vranges = [None] * len(axes_row)
    for ax, val, title, vr in zip(axes_row, values_list, titles, vranges):
        kw = {}
        if vr is not None:
            kw["vmin"], kw["vmax"] = vr
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, cmap="viridis", s=1, **kw)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.grid(True, linewidth=0.3)
        scs.append(sc)
    return scs


def add_colorbars(fig, axes_row, scs):
    for ax, sc in zip(axes_row, scs):
        fig.colorbar(sc, ax=ax, location="right", shrink=0.6, pad=0.05)


def make_gif(fig, axes_flat, update_fn, path):
    """Create animation and save as GIF using PillowWriter."""
    anim = FuncAnimation(fig, update_fn, frames=len(snapshots), blit=False)
    anim.save(path, writer=writer)
    plt.close(fig)
    print(f"Saved {path}  ({len(snapshots)} frames, {GIF_FPS} fps)")


def update_pred(frame):
    step_num, pred_np, _ = snapshots[frame]
    fig.suptitle(f"Predictions  —  step {step_num}", fontsize=9)
    for i, sc in enumerate(scs):
        sc.set_array(pred_np[:, i])


def update_err(frame):
    step_num, _, rel_l1 = snapshots[frame]
    fig.suptitle(f"Relative L1 error  —  step {step_num}", fontsize=9)
    for i, sc in enumerate(scs):
        sc.set_array(rel_l1[:, i])


def update_combined(frame):
    step_num, pred_np, rel_l1 = snapshots[frame]
    fig.suptitle(f"step {step_num}", fontsize=10)
    for i in range(3):
        scs_p[i].set_array(pred_np[:, i])
        scs_e[i].set_array(rel_l1[:, i])


model_name = "FBPINN_full_5686"
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
CHECKPOINT_DIR = "../checkpoints" + f"/{model_name}"
LAST_CHECKPOINT = 100_000
CHECKPOINT_STEP = 5000  # use every k-th checkpoint (sorted by step number)
POINT_STEP = 10  # subsample 1-in-N points for plotting
GIF_FPS = 2
OUTPUT_DIR = "../gifs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch on cuda:", torch.cuda.is_available())
domain, hole = prepare_geometry()
block_size = (0.25, 0.25)
overlap = (0.08, 0.08)
bbox_left = (-block_size[0] / 2, -block_size[1] / 2)
bbox_right = (2000 * scale + block_size[0] / 2, 1200 * scale + block_size[1] / 2)
# block_scales = {"vx": 0.00137, "vy": 0.00142, "pressure": 1.46e-5}
# block_shifts = {"vx": 0.007, "vy": -3.5e-6, "pressure": -2.93e-6}
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
    points_per_block=200,
    eps_full=1e-3,
    holes=[hole],
    device=device,
)
# dec.remove_redundant_blocks(samples_per_block=4000, tol=0.001, verbose=False)
plot_decomposition2d(dec.blocks, polygon_vertices=domain, holes=[hole], figsize=(12, 4))
print(f"Decomposition has {len(dec.blocks)} blocks")

model_config = {
    "input_size": 2,
    "output_size": 3,
    "activation_func": ["tanh", "tanh", "tanh", "linear"],
    "models_size": [16, 16, 16],
    "device": device,
    "weight": torch.nn.init.xavier_uniform_,
    "biases": torch.nn.init.zeros_,
}
pde = CylinderViscid(cylinder=hole, scale=scale, device=device, path_to_data="../")
nn = FBPINN(
    **model_config,
    physic_loss=pde.phys_loss,
    boundary_loss=pde.sub_losses,
    decomposition=dec,
)
nn.to(device)

x_all = pde.val_input
y_all = torch.tensor(pde.solution(x_all), device=device)
x_plot_dev = torch.tensor(x_all[::POINT_STEP], device=device)
y_plot = y_all[::POINT_STEP].detach().cpu()
xy = x_plot_dev.detach().cpu().numpy()


nn.eval()
snapshots = []  # list of (step_num, pred_np, rel_l1_np)
for ckpt_num in range(0, LAST_CHECKPOINT, CHECKPOINT_STEP):
    path = CHECKPOINT_DIR + f"/fbpinn123_{ckpt_num}.weights.h5"
    nn.load_weights(path)
    with torch.no_grad():
        pred = nn(x_plot_dev).detach().cpu()
    rel_l1 = (
        torch.abs(y_plot - pred) / torch.maximum(torch.abs(y_plot), torch.tensor(1e-8))
    ).numpy()
    snapshots.append((ckpt_num, pred.numpy(), rel_l1))

p_min = min(
    list(snapshots[i][1][:, 0].min() for i in range(LAST_CHECKPOINT // CHECKPOINT_STEP))
    + [y_plot[:, 0].min().item()]
)
p_max = max(
    list(snapshots[i][1][:, 0].max() for i in range(LAST_CHECKPOINT // CHECKPOINT_STEP))
    + [y_plot[:, 0].max().item()]
)
vx_min = min(
    list(snapshots[i][1][:, 1].min() for i in range(LAST_CHECKPOINT // CHECKPOINT_STEP))
    + [y_plot[:, 1].min().item()]
)
vx_max = max(
    list(snapshots[i][1][:, 1].max() for i in range(LAST_CHECKPOINT // CHECKPOINT_STEP))
    + [y_plot[:, 1].max().item()]
)
vy_min = min(
    list(snapshots[i][1][:, 2].min() for i in range(LAST_CHECKPOINT // CHECKPOINT_STEP))
    + [y_plot[:, 2].min().item()]
)
vy_max = max(
    list(snapshots[i][1][:, 2].max() for i in range(LAST_CHECKPOINT // CHECKPOINT_STEP))
    + [y_plot[:, 2].max().item()]
)

truth_np = y_plot.numpy()
writer = PillowWriter(fps=GIF_FPS)

LABELS = ["pressure", "vx", "vy"]
# fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), tight_layout=True)
# scs = scatter_grid(axes, xy, [truth_np[:, i] for i in range(3)],
#                    [f"Predicted {l}" for l in LABELS])
# add_colorbars(fig, axes, scs)
# make_gif(fig, axes, update_pred, os.path.join(OUTPUT_DIR, "predictions.gif"))
#
# fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), tight_layout=True)
_, pred_np, rel_l1_0 = snapshots[0]
# scs = scatter_grid(axes, xy, [rel_l1_0[:, i] for i in range(3)],
#                    [f"Rel L1 {l}" for l in LABELS], vranges=(0, 2))
# add_colorbars(fig, axes, scs)
# make_gif(fig, axes, update_err, os.path.join(OUTPUT_DIR, "errors.gif"))

fig, axes = plt.subplots(3, 3, figsize=(13, 10))
scs_p = scatter_grid(
    axes[0],
    xy,
    [pred_np[:, i] for i in range(3)],
    [f"Predicted {l}" for l in LABELS],
    vranges=[(p_min, p_max), (vx_min, vx_max), (vy_min, vy_max)],
)
scs_t = scatter_grid(
    axes[1],
    xy,
    [truth_np[:, i] for i in range(3)],
    [f"Truth {l}" for l in LABELS],
    vranges=[(p_min, p_max), (vx_min, vx_max), (vy_min, vy_max)],
)
scs_e = scatter_grid(
    axes[2],
    xy,
    [rel_l1_0[:, i] for i in range(3)],
    [f"Rel L1 {l}" for l in LABELS],
    vranges=[(0, 2), (0, 2), (0, 2)],
)
add_colorbars(fig, axes[0], scs_p)
add_colorbars(fig, axes[1], scs_t)
cbar = fig.colorbar(scs_e[0], ax=axes[2, :], location="right", shrink=0.8, pad=0.05)
make_gif(
    fig, axes, update_combined, os.path.join(OUTPUT_DIR, f"{model_name}_training.gif")
)
