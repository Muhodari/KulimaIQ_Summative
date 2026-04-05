#!/usr/bin/env python3
"""
Generalization test: evaluate saved agents under harder stochastic dynamics
(KulimaProductionEnv dynamics_shift > 1).

Run from project root (requires trained models from main.py paths):
    python analysis/generalization_eval.py --algo dqn --episodes 20

Prints mean ± std return for baseline (shift=1.0) vs shifted (shift=1.25).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from environment.kulima_production_env import KulimaProductionEnv
from training.env_factory import PRODUCTION_MAX_STEPS


def load_best(algo: str) -> str:
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
    with open(p) as f:
        d = json.load(f)
    if algo == "reinforce":
        return str(d["checkpoint"])
    return str(d["best_model"])


def eval_sb3(algo: str, path: str, episodes: int, shift: float, seed0: int) -> List[float]:
    from stable_baselines3 import A2C, DQN, PPO

    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
    model = cls.load(path)
    env = KulimaProductionEnv(render_mode=None, max_steps=PRODUCTION_MAX_STEPS, dynamics_shift=shift)
    rets: List[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        g = 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(a))
            g += float(r)
            done = term or trunc
        rets.append(g)
    env.close()
    return rets


def eval_reinforce(path: str, episodes: int, shift: float, seed0: int) -> List[float]:
    import torch
    from training.reinforce_training import PolicyNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(path, map_location=device)
    cfg = data["cfg"]
    env = KulimaProductionEnv(render_mode=None, max_steps=PRODUCTION_MAX_STEPS, dynamics_shift=shift)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNet(obs_dim, n_actions, hidden=int(cfg["hidden"])).to(device)
    policy.load_state_dict(data["policy"])
    policy.eval()
    rets: List[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        g = 0.0
        while not done:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(o)
                a = torch.argmax(logits, dim=-1).item()
            obs, r, term, trunc, _ = env.step(int(a))
            g += float(r)
            done = term or trunc
        rets.append(g)
    env.close()
    return rets


def summarize(xs: List[float]) -> str:
    a = np.asarray(xs, dtype=np.float64)
    return f"mean={a.mean():.3f} std={a.std():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--shift", type=float, default=1.25)
    ap.add_argument("--seed0", type=int, default=5000)
    args = ap.parse_args()

    path = load_best(args.algo)
    if args.algo == "reinforce":
        base = eval_reinforce(path, args.episodes, 1.0, args.seed0)
        shifted = eval_reinforce(path, args.episodes, args.shift, args.seed0 + 10_000)
    else:
        base = eval_sb3(args.algo, path, args.episodes, 1.0, args.seed0)
        shifted = eval_sb3(args.algo, path, args.episodes, args.shift, args.seed0 + 10_000)

    print(f"Algo={args.algo} model={path}")
    print(f"Baseline dynamics_shift=1.0:  {summarize(base)}")
    print(f"Shifted  dynamics_shift={args.shift}: {summarize(shifted)}")


if __name__ == "__main__":
    main()
