train_config = {
    "epochs": 100_000,
    "start_epoch": 0,
    "patience": 4_000_000,
    "eval_interval": 100,
    "log_interval": 1000,
    "mode": "layer",
}

scheduler_config = {
    "mode": "min",
    "factor": 0.8,
    "patience": 50,
    "warmup_epochs": 10,
    "warmup_start_factor": 0.1,
}
