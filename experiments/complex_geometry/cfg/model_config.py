import torch

embedding_config = {
    "input_dim": 2,
    "output_dim": 2,
}
model_config = {
    "input_size": 2,
    "output_size": 3,
    "activation_func": ["tanh", "tanh", "linear"],
    "models_size": [32, 32],
    "weight": torch.nn.init.xavier_uniform_,
    "biases": torch.nn.init.zeros_,
    "affected_radius": 1,
}

compile_config = {
    "rate": 1e-4,
    "optimizer": "AdamW",
}
