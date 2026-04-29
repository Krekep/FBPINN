import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import random
from experiments.complex_geometry.functions.poisson_3d import Poisson3D
from geofbpinn.geometry import DecompositionND, RectangleDomain
from geofbpinn.networks.schedulers.layer import BaseLayerScheduler
from geofbpinn.networks.schedulers.loss import AdaptiveLossScheduler, LossScheduler
from geofbpinn.networks.topology.fbpinn.model import FBPINN
from geofbpinn.networks.topology.fbpinn.trainer import train_fbpinn
from geofbpinn.networks.schedulers.lr import WarmupReduceLROnPlateau
import mlflow

print("Torch on cuda", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = (0.4, 0.4, 0.4)
overlap = (0.12, 0.12, 0.12)
bbox_left = (-block_size[0] / 4, -block_size[1] / 4, -block_size[2] / 4)
bbox_right = (1 + block_size[0] / 4, 1 + block_size[1] / 4, 1 + block_size[2] / 4)
points_per_block = 200
eps_full = 1e-3
lr = 1e-4
block_scales = {
    "u": 1,
}
block_shifts = {
    "u": 0,
}
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
    points_per_block=points_per_block,
    device=device,
)
# dec.remove_redundant_blocks(samples_per_block=2000, tol=0.0001, verbose=False)
# plot_decomposition(dec.blocks, polygon_vertices=domain, holes=[hole], figsize=(12, 4), savepath="decomposition_airfoil.png")
print(f"Decomposition has {len(dec.blocks)} blocks")

model_config = {
    "input_size": 3,
    "output_size": 1,
    "activation_func": ["tanh", "tanh", "tanh", "linear"],
    "models_size": [32, 32, 32],
    "device": device,
    "weight": torch.nn.init.xavier_uniform_,
    "biases": torch.nn.init.zeros_,
}

pde = Poisson3D(device=device)
phys_loss = pde.phys_loss
which = pde.description

mlflow.set_experiment("FBPINN Poisson")
run_id = random.randint(1, 10000)
run_name = f"FBPINN_full_{run_id}"
mlflow.start_run(run_name=run_name, log_system_metrics=True)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("Class", pde.equation_class)
mlflow.log_param("left_bound", bbox_left)
mlflow.log_param("right_bound", bbox_right)
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

x = pde.val_input[::10]
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
    "loss_weights": [1, 1, 1, 1, 1, 1, 1000],
}
loss_scheduler = LossScheduler(**loss_scheduler_config)
mlflow.log_param("Loss scheduler", "LossScheduler")
mlflow.log_params(loss_scheduler_config)

train_config = {
    "epochs": 100000,
    "start_epoch": 0,
    "patience": 4_000_000,
    "eval_interval": 100,
    "log_interval": 1000,
    "mode": "layer",
    "layer_scheduler": layer_scheduler,
    "loss_scheduler": loss_scheduler,
    "path_to_ckpt": f"./checkpoints/{run_name}",
}
mlflow.log_params(train_config)
scheduler = WarmupReduceLROnPlateau(
    nn.optimizer,
    mode="min",
    factor=0.8,
    patience=50,
    warmup_epochs=10,
    warmup_start_factor=0.1,
)
train_fbpinn(
    **train_config,
    fbpinn=nn,
    callbacks=[],
    val_truth=y,
    val_input=x,
    png_salt=str("123"),
    lr_scheduler=scheduler,
)
nn.save_weights(f"./checkpoints/{run_name}/poisson_last.weights.h5")

mlflow.end_run()
