import torch

env_config = {
    "Decomposition": "Decomposition2DPolygon",
    "model_name": "FBPINN_full",
    "embedding": "Base",
    "layer_scheduler": "Base",
    "loss_scheduler": "Base",
    "scheduler": "WarmupReduceLROnPlateau",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_name": "FBPINN",
    "checkpoint_dir": "./checkpoints",
    "experiment_name": "Viscid Cylinder",
    "random_seed": 42,
}
