#!/usr/bin/env python3
"""
Policy-gradient family via Stable-Baselines3: PPO and A2C (Actor–Critic).
Each algorithm runs 10 hyperparameter configurations (see hyperparams.py).

Usage:
    python training/pg_training.py --algo ppo
    python training/pg_training.py --algo a2c
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from training.env_factory import make_monitored_env
from training.hyperparams import A2C_SWEEPS, PPO_SWEEPS, TOTAL_TIMESTEPS
from training.metrics_callback import SummativeMetricsJSONLCallback


def _progress_bar_ok() -> bool:
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401

        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "a2c"], required=True)
    args = ap.parse_args()

    sweeps = PPO_SWEEPS if args.algo == "ppo" else A2C_SWEEPS
    out_root = os.path.join(ROOT, "models", "pg", args.algo)
    os.makedirs(out_root, exist_ok=True)
    summaries = []

    Algo = PPO if args.algo == "ppo" else A2C

    for i, cfg in enumerate(sweeps):
        run_dir = os.path.join(out_root, f"run_{i}")
        os.makedirs(run_dir, exist_ok=True)
        train_env = DummyVecEnv([make_monitored_env(seed=20 + i)])
        eval_env = DummyVecEnv([make_monitored_env(seed=888 + i)])

        model = Algo("MlpPolicy", train_env, verbose=0, tensorboard_log=os.path.join(run_dir, "tb"), **cfg)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=max(2000, TOTAL_TIMESTEPS // 20),
            deterministic=True,
            render=False,
        )
        metrics_cb = SummativeMetricsJSONLCallback(
            os.path.join(run_dir, "metrics.jsonl"),
            log_every=max(256, TOTAL_TIMESTEPS // 200),
        )
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=CallbackList([eval_cb, metrics_cb]),
            progress_bar=_progress_bar_ok(),
        )
        best_path = os.path.join(run_dir, "best_model.zip")
        if not os.path.isfile(best_path):
            model.save(os.path.join(run_dir, "final_model.zip"))
            best_path = os.path.join(run_dir, "final_model.zip")

        mean_reward = None
        eval_file = os.path.join(run_dir, "evaluations.npz")
        if os.path.isfile(eval_file):
            import numpy as np

            data = np.load(eval_file, allow_pickle=True)
            if "results" in data:
                mean_reward = float(np.mean(data["results"][-5:]))

        rec = {"algo": args.algo, "run_id": i, "cfg": cfg, "best_model": best_path, "mean_eval": mean_reward}
        summaries.append(rec)
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(rec, f, indent=2)

        train_env.close()
        eval_env.close()

    scored = [s for s in summaries if s.get("mean_eval") is not None]
    best = max(scored, key=lambda x: x["mean_eval"]) if scored else summaries[-1]
    with open(os.path.join(out_root, f"{args.algo}_best.json"), "w") as f:
        json.dump(best, f, indent=2)
    print(f"{args.algo.upper()} best:", best)


if __name__ == "__main__":
    main()
