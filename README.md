[![Check code style](https://github.com/Krekep/FBPINN/actions/workflows/code_style.yml/badge.svg)](https://github.com/Krekep/FBPINN/actions/workflows/code_style.yml)
[![Run tests](https://github.com/Krekep/FBPINN/actions/workflows/tests.yml/badge.svg)](https://github.com/Krekep/FBPINN/actions/workflows/tests.yml)

# FBPINN

FBPINNs are described in detail here:
*[Finite Basis Physics-Informed Neural Networks (FBPINNs): a scalable domain decomposition approach for solving differential equations](https://link.springer.com/article/10.1007/s10444-023-10065-9),
B. Moseley, T. Nissen-Meyer and A. Markham, Jul 2023 Advances in Computational Mathematics*.

**See the [original repository](https://github.com/benmoseley/FBPINNs).**

# GeoFBPINN

FBPINN with automatic domain decomposition.
Supports N-dimensional rectangular domains and arbitrary 2D polygon domains (including holes).

---

## Requirements

- Python 3.10+
- PyTorch 2.8+

## Installation

```bash
pip install geofbpinn
```

Or from source:

```bash
git clone https://github.com/Krekep/FBPINN.git
cd FBPINN
pip install -e .
```

For development tools (pytest, black, pre-commit):

```bash
pip install -e ".[dev]"
```

---

## Usage

More examples look in `./examples/`.

### 1. Define domain decomposition

**Rectangular N-dimensional domain:**

```python
from geofbpinn.geometry.base_decomposition import RectangleDomain
from geofbpinn.geometry.decomposition import DecompositionND

domain = RectangleDomain(left_corner=[0.0, 0.0], right_corner=[1.0, 1.0])

decomp = DecompositionND(
    domain=domain,
    bbox_left=[0.0, 0.0],
    bbox_right=[1.0, 1.0],
    overlap=[0.1, 0.1],
    block_size=[0.4, 0.4],
    block_scales=[1.0, 1.0],
    block_shift=[0.0, 0.0],
    points_per_block=200,
    device="cuda",
)
```

**2D polygon domain (with optional holes):**

```python
from geofbpinn.geometry.polygon_decomposition import Decomposition2DPolygon

polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
hole    = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]

decomp = Decomposition2DPolygon(
    polygon_vertices=polygon,
    bbox_left=(0.0, 0.0),
    bbox_right=(1.0, 1.0),
    block_scales=[1.0, 1.0],
    block_shift=[0.0, 0.0],
    block_size=(0.4, 0.4),
    overlap=(0.1, 0.1),
    points_per_block=200,
    holes=[hole],
    device="cuda",
)

# Optionally remove blocks fully covered by neighbours
decomp.remove_redundant_blocks(samples_per_block=400, tol=0.01)
```

### 2. Build and train FBPINN

```python
import torch
from geofbpinn.networks.topology.fbpinn.model import FBPINN
from geofbpinn.networks.topology.fbpinn.fbpinn_train import layer_train
from geofbpinn.networks.schedulers.layer import BaseLayerScheduler
from geofbpinn.networks.schedulers.loss import LossScheduler

def pde_loss(fbpinn, data, active_models):
    # residual of your PDE
    ...

def bc_loss(fbpinn, data, active_models):
    # boundary condition residual
    ...

bc_polygon = [(0, 0), (1, 0), (1, 0.01), (0, 0.01)]  # bottom edge

model = FBPINN(
    input_size=2,
    output_size=1,
    decomposition=decomp,
    physic_loss=pde_loss,
    boundary_loss=[(bc_loss, bc_polygon)],
    activation_func="tanh",
    models_size=[32, 32],
    device="cuda",
)

layer_train(
    fbpinn=model,
    epochs=5000,
    val_input=val_x,
    val_truth=val_y,
    layer_scheduler=BaseLayerScheduler(...),
    loss_scheduler=LossScheduler(...),
    path_to_ckpt="checkpoints/",
    log_interval=500,
)
```

Training is tracked automatically via **MLflow**. Start the UI with:

```bash
mlflow ui
```

---

## Project structure

```
geofbpinn/
├── geometry/
│   ├── base_decomposition.py      # Block, RectangleDomain, BaseDecomposition
│   ├── decomposition.py           # DecompositionND
│   ├── polygon_decomposition.py   # Decomposition2DPolygon, PolygonBlock
│   ├── geometry.py                # Polygon utilities (clip, area, sampling)
│   └── plot.py                    # Decomposition visualisation
└── networks/
    ├── topology/fbpinn/
    │   ├── model.py               # FBPINN
    │   ├── fbpinn_train.py
    │   └── trainer.py             # Main train function
    ├── schedulers/                # Layer, loss, LR schedulers
    ├── layers/dense.py
    ├── activations.py
    ├── optimizers.py
    ├── metrics.py
    └── losses.py
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).
