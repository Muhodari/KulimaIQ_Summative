"""
Hyperparameter grids for extensive tuning (≥10 runs per algorithm).
Adjust TOTAL_TIMESTEPS for longer production training.
"""

# Short default for CI / classroom demos; increase to 200k–1M for stronger policies
TOTAL_TIMESTEPS = int(__import__("os").environ.get("RL_TIMESTEPS", "40000"))

# --- DQN (10 runs) ---
DQN_SWEEPS = [
    {"learning_rate": 1e-4, "buffer_size": 50_000, "batch_size": 64, "gamma": 0.99, "train_freq": 4, "target_update_interval": 1_000, "exploration_fraction": 0.2, "exploration_final_eps": 0.05},
    {"learning_rate": 3e-4, "buffer_size": 100_000, "batch_size": 128, "gamma": 0.99, "train_freq": 4, "target_update_interval": 2_000, "exploration_fraction": 0.3, "exploration_final_eps": 0.08},
    {"learning_rate": 5e-4, "buffer_size": 50_000, "batch_size": 32, "gamma": 0.98, "train_freq": 2, "target_update_interval": 500, "exploration_fraction": 0.15, "exploration_final_eps": 0.03},
    {"learning_rate": 1e-4, "buffer_size": 200_000, "batch_size": 256, "gamma": 0.995, "train_freq": 8, "target_update_interval": 4_000, "exploration_fraction": 0.25, "exploration_final_eps": 0.06},
    {"learning_rate": 2e-4, "buffer_size": 80_000, "batch_size": 64, "gamma": 0.99, "train_freq": 4, "target_update_interval": 1_500, "exploration_fraction": 0.35, "exploration_final_eps": 0.1},
    {"learning_rate": 4e-4, "buffer_size": 60_000, "batch_size": 64, "gamma": 0.97, "train_freq": 4, "target_update_interval": 800, "exploration_fraction": 0.2, "exploration_final_eps": 0.04},
    {"learning_rate": 1e-3, "buffer_size": 40_000, "batch_size": 128, "gamma": 0.99, "train_freq": 4, "target_update_interval": 2_000, "exploration_fraction": 0.4, "exploration_final_eps": 0.12},
    {"learning_rate": 5e-5, "buffer_size": 150_000, "batch_size": 64, "gamma": 0.995, "train_freq": 4, "target_update_interval": 3_000, "exploration_fraction": 0.18, "exploration_final_eps": 0.05},
    {"learning_rate": 3e-4, "buffer_size": 100_000, "batch_size": 64, "gamma": 0.99, "train_freq": 1, "target_update_interval": 1_000, "exploration_fraction": 0.22, "exploration_final_eps": 0.07},
    {"learning_rate": 2.5e-4, "buffer_size": 120_000, "batch_size": 96, "gamma": 0.988, "train_freq": 4, "target_update_interval": 1_200, "exploration_fraction": 0.28, "exploration_final_eps": 0.065},
]

# --- PPO (10 runs) ---
PPO_SWEEPS = [
    {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5},
    {"learning_rate": 1e-4, "n_steps": 2048, "batch_size": 128, "gamma": 0.995, "gae_lambda": 0.92, "clip_range": 0.15, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5},
    {"learning_rate": 5e-4, "n_steps": 512, "batch_size": 32, "gamma": 0.98, "gae_lambda": 0.97, "clip_range": 0.25, "ent_coef": 0.02, "vf_coef": 0.3, "max_grad_norm": 1.0},
    {"learning_rate": 2e-4, "n_steps": 1024, "batch_size": 256, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.005, "vf_coef": 0.6, "max_grad_norm": 0.3},
    {"learning_rate": 4e-4, "n_steps": 1536, "batch_size": 96, "gamma": 0.997, "gae_lambda": 0.9, "clip_range": 0.18, "ent_coef": 0.0, "vf_coef": 0.45, "max_grad_norm": 0.7},
    {"learning_rate": 2.5e-4, "n_steps": 1024, "batch_size": 64, "gamma": 0.99, "gae_lambda": 0.98, "clip_range": 0.12, "ent_coef": 0.03, "vf_coef": 0.5, "max_grad_norm": 0.5},
    {"learning_rate": 6e-4, "n_steps": 256, "batch_size": 64, "gamma": 0.95, "gae_lambda": 0.99, "clip_range": 0.3, "ent_coef": 0.01, "vf_coef": 0.4, "max_grad_norm": 1.5},
    {"learning_rate": 1e-3, "n_steps": 512, "batch_size": 128, "gamma": 0.99, "gae_lambda": 0.93, "clip_range": 0.22, "ent_coef": 0.0, "vf_coef": 0.55, "max_grad_norm": 0.4},
    {"learning_rate": 8e-5, "n_steps": 2048, "batch_size": 64, "gamma": 0.995, "gae_lambda": 0.96, "clip_range": 0.2, "ent_coef": 0.015, "vf_coef": 0.5, "max_grad_norm": 0.6},
    {"learning_rate": 3.5e-4, "n_steps": 768, "batch_size": 48, "gamma": 0.992, "gae_lambda": 0.94, "clip_range": 0.17, "ent_coef": 0.008, "vf_coef": 0.52, "max_grad_norm": 0.55},
]

# --- A2C (10 runs) — SB3 A2C ---
A2C_SWEEPS = [
    {"learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5},
    {"learning_rate": 3e-4, "n_steps": 5, "gamma": 0.98, "gae_lambda": 1.0, "ent_coef": 0.01, "vf_coef": 0.4, "max_grad_norm": 0.7},
    {"learning_rate": 1e-3, "n_steps": 5, "gamma": 0.995, "gae_lambda": 1.0, "ent_coef": 0.02, "vf_coef": 0.6, "max_grad_norm": 1.0},
    {"learning_rate": 5e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.005, "vf_coef": 0.5, "max_grad_norm": 0.3},
    {"learning_rate": 2e-4, "n_steps": 5, "gamma": 0.97, "gae_lambda": 1.0, "ent_coef": 0.0, "vf_coef": 0.45, "max_grad_norm": 0.5},
    {"learning_rate": 9e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.03, "vf_coef": 0.35, "max_grad_norm": 1.2},
    {"learning_rate": 4e-4, "n_steps": 5, "gamma": 0.996, "gae_lambda": 1.0, "ent_coef": 0.01, "vf_coef": 0.55, "max_grad_norm": 0.45},
    {"learning_rate": 6e-4, "n_steps": 5, "gamma": 0.985, "gae_lambda": 1.0, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.8},
    {"learning_rate": 1.2e-3, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.015, "vf_coef": 0.42, "max_grad_norm": 0.65},
    {"learning_rate": 2.5e-4, "n_steps": 5, "gamma": 0.993, "gae_lambda": 1.0, "ent_coef": 0.008, "vf_coef": 0.48, "max_grad_norm": 0.55},
]

# --- REINFORCE (10 runs) — custom PyTorch; not all SB3 params apply ---
REINFORCE_SWEEPS = [
    {"lr": 1e-3, "hidden": 128, "gamma": 0.99, "entropy_coef": 0.0},
    {"lr": 3e-4, "hidden": 256, "gamma": 0.99, "entropy_coef": 0.01},
    {"lr": 5e-4, "hidden": 128, "gamma": 0.995, "entropy_coef": 0.0},
    {"lr": 1e-4, "hidden": 64, "gamma": 0.98, "entropy_coef": 0.02},
    {"lr": 2e-3, "hidden": 128, "gamma": 0.97, "entropy_coef": 0.005},
    {"lr": 4e-4, "hidden": 192, "gamma": 0.99, "entropy_coef": 0.0},
    {"lr": 6e-4, "hidden": 128, "gamma": 0.992, "entropy_coef": 0.015},
    {"lr": 8e-4, "hidden": 256, "gamma": 0.985, "entropy_coef": 0.0},
    {"lr": 2.5e-4, "hidden": 96, "gamma": 0.99, "entropy_coef": 0.012},
    {"lr": 7e-4, "hidden": 160, "gamma": 0.988, "entropy_coef": 0.003},
]
