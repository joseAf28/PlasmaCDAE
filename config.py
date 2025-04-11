import numpy as np

config_set = {
    "seed": 42,
    "ratio_test_val_train": 0.15,
    "data_path": "data_3000_points.txt",
    "batch_size": 32,
    "device": "cpu",
}

config_cade = {
    "x_dim": 3,
    "y_dim": 17,
    "hidden_dim": 128,
    "latent_dim": 64,
    "noise_embed_dim": 5,
    "p_noise": 0.95,
    "min_noise": 1e-4,
    "max_noise": 0.3,
    "noise_schedule_dim": 10,
    "noise_dim": 5,
    "num_epochs": 1100,
    "lr": 1e-3,
    "lambda_sparse": 1e-5,
}


config_mapping = {
    "hidden_dim": 200,
    "lambda_reg": 1e-5,
    "num_epochs": 770
}


config_refinement = {
    "num_iters_per_level": 1000,
    "step_size": 1e-3,
    "eps_convergence": 1e-3,
    "eps_clip": 5e-2
}