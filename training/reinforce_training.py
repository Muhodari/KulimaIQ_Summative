#!/usr/bin/env python3
"""
Vanilla REINFORCE (Monte Carlo policy gradient) for KulimaFarmMissionEnv.
Not included in Stable-Baselines3; implemented in PyTorch for fair comparison with SB3 agents.

Run from project root:
    python training/reinforce_training.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from environment.kulima_production_env import KulimaProductionEnv
from training.env_factory import PRODUCTION_MAX_STEPS
from training.hyperparams import REINFORCE_SWEEPS, TOTAL_TIMESTEPS


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_episode(
    env: KulimaProductionEnv,
    policy: PolicyNet,
    device: torch.device,
    max_steps: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], float]:
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    rewards: List[float] = []
    obs, _ = env.reset()
    done = False
    steps = 0
    ep_return = 0.0
    while not done and steps < max_steps:
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(o)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        obs, r, term, trunc, _ = env.step(int(action.item()))
        rewards.append(float(r))
        ep_return += float(r)
        done = term or trunc
        steps += 1
    return log_probs, entropies, rewards, ep_return


def train_run(
    run_id: int,
    cfg: dict,
    episodes_budget: int,
    out_dir: str,
    seed: int = 0,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = KulimaProductionEnv(render_mode=None, max_steps=PRODUCTION_MAX_STEPS)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_dim, n_actions, hidden=int(cfg["hidden"])).to(device)
    opt = optim.Adam(policy.parameters(), lr=float(cfg["lr"]))
    gamma = float(cfg["gamma"])
    entropy_coef = float(cfg["entropy_coef"])

    returns_history: List[float] = []
    entropy_history: List[float] = []
    metrics_path = os.path.join(out_dir, f"reinforce_run_{run_id}_metrics.jsonl")
    open(metrics_path, "w").close()
    for ep in range(episodes_budget):
        log_probs, entropies, rewards, G = collect_episode(env, policy, device, env.max_steps)
        returns_history.append(G)
        if entropies:
            mean_h = float(torch.stack(list(entropies)).mean().item())
            entropy_history.append(mean_h)
        else:
            mean_h = 0.0
        with open(metrics_path, "a") as mf:
            mf.write(
                json.dumps(
                    {"episode": ep, "return": G, "mean_entropy": mean_h, "n_steps": len(rewards)}
                )
                + "\n"
            )
        if len(rewards) == 0:
            continue
        # Discounted returns (Monte Carlo)
        R = 0.0
        discounted: List[float] = []
        for r in reversed(rewards):
            R = r + gamma * R
            discounted.append(R)
        discounted.reverse()
        returns = torch.tensor(discounted, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for t in range(len(log_probs)):
            loss -= log_probs[t] * returns[t]
            loss -= entropy_coef * entropies[t]
        loss = loss / len(log_probs)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()

    env.close()

    ckpt = os.path.join(out_dir, f"reinforce_run_{run_id}.pt")
    torch.save({"policy": policy.state_dict(), "cfg": cfg}, ckpt)

    summary = {
        "algorithm": "REINFORCE",
        "run_id": run_id,
        "cfg": cfg,
        "episodes": episodes_budget,
        "mean_return_last_50": float(np.mean(returns_history[-50:])) if returns_history else 0.0,
        "mean_entropy_last_50": float(np.mean(entropy_history[-50:])) if entropy_history else 0.0,
        "checkpoint": ckpt,
        "metrics_jsonl": metrics_path,
    }
    with open(os.path.join(out_dir, f"reinforce_run_{run_id}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    out_root = os.path.join(ROOT, "models", "pg")
    os.makedirs(out_root, exist_ok=True)
    # Scale episodes to approximate similar wall-clock order as SB3 timesteps (rough heuristic)
    episodes_per_run = max(200, TOTAL_TIMESTEPS // 200)

    all_summaries = []
    for i, cfg in enumerate(REINFORCE_SWEEPS):
        print(f"REINFORCE run {i+1}/{len(REINFORCE_SWEEPS)}", cfg)
        s = train_run(i, cfg, episodes_per_run, out_root, seed=100 + i)
        all_summaries.append(s)

    best = max(all_summaries, key=lambda x: x["mean_return_last_50"])
    with open(os.path.join(out_root, "reinforce_best.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("Best REINFORCE:", best)


if __name__ == "__main__":
    main()
