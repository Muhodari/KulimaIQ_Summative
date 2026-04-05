#!/usr/bin/env python3
"""
Entry point: run the best-performing saved model (after training).

Examples:
    python main.py --algo dqn                    # opens Pygame UI (default)
    python main.py --algo ppo --episodes 10
    python main.py --algo dqn --render none    # headless, text only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from environment.kulima_production_env import KulimaProductionEnv
from training.env_factory import PRODUCTION_MAX_STEPS


def load_best_path(algo: str) -> str:
    if algo == "dqn":
        p = os.path.join(ROOT, "models", "dqn", "dqn_best.json")
    elif algo == "ppo":
        p = os.path.join(ROOT, "models", "pg", "ppo", "ppo_best.json")
    elif algo == "a2c":
        p = os.path.join(ROOT, "models", "pg", "a2c", "a2c_best.json")
    elif algo == "reinforce":
        p = os.path.join(ROOT, "models", "pg", "reinforce_best.json")
    else:
        raise ValueError(algo)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing {p}. Train the algorithm first.")
    with open(p) as f:
        data = json.load(f)
    if algo == "reinforce":
        return data["checkpoint"]
    return data["best_model"]


def run_sb3(algo: str, model_path: str, episodes: int, render: str | None):
    from stable_baselines3 import A2C, DQN, PPO

    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
    model = cls.load(model_path)
    env = KulimaProductionEnv(render_mode=render, max_steps=PRODUCTION_MAX_STEPS)
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        if render:
            env.render()
        done = False
        ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            ret += r
            done = term or trunc
            if render:
                env.render()
        print(f"Episode {ep+1} return={ret:.2f} info={info}")
    env.close()


def run_reinforce(ckpt: str, episodes: int, render: str | None):
    import torch
    from training.reinforce_training import PolicyNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(ckpt, map_location=device)
    cfg = data["cfg"]
    env = KulimaProductionEnv(render_mode=render, max_steps=PRODUCTION_MAX_STEPS)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNet(obs_dim, n_actions, hidden=int(cfg["hidden"])).to(device)
    policy.load_state_dict(data["policy"])
    policy.eval()

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        if render:
            env.render()
        done = False
        ret = 0.0
        while not done:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(o)
                action = torch.argmax(logits, dim=-1).item()
            obs, r, term, trunc, info = env.step(int(action))
            ret += r
            done = term or trunc
            if render:
                env.render()
        print(f"Episode {ep+1} return={ret:.2f} info={info}")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default="dqn")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument(
        "--render",
        choices=["human", "none", "rgb_array"],
        default="human",
        help="human = Pygame window (default); none = terminal only",
    )
    args = ap.parse_args()
    render = None if args.render == "none" else args.render

    path = load_best_path(args.algo)
    print("Loading:", path)
    if args.algo == "reinforce":
        run_reinforce(path, args.episodes, render)
    else:
        run_sb3(args.algo, path, args.episodes, render)


if __name__ == "__main__":
    main()
