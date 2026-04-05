#!/usr/bin/env python3
"""
Build summative figures from logged training artifacts:
  - Eval / convergence curves (EvalCallback evaluations.npz)
  - DQN training loss (metrics.jsonl)
  - Policy-gradient entropy-related scalars (metrics.jsonl)
  - REINFORCE episode returns + entropy (reinforce_run_*_metrics.jsonl)
  - Bar chart: mean eval reward per hyperparameter run

Run from project root after training:
    python analysis/plot_summative.py

Outputs go to results/figures/ (created automatically).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "results", "figures")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pick_scalar_row(rows: List[Dict[str, Any]], candidates: Tuple[str, ...]) -> Tuple[List[float], List[float], str]:
    xs: List[float] = []
    ys: List[float] = []
    key_used = ""
    for row in rows:
        t = float(row.get("timestep", row.get("n_calls", 0)))
        for k in candidates:
            if k in row:
                xs.append(t)
                ys.append(float(row[k]))
                key_used = k
                break
    return xs, ys, key_used


def plot_eval_convergence() -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=False)
    fig.suptitle("Eval reward vs training progress (EvalCallback)", fontsize=13)

    specs = [
        ("DQN", os.path.join(ROOT, "models", "dqn", "run_*", "evaluations.npz"), axes[0, 0]),
        ("PPO", os.path.join(ROOT, "models", "pg", "ppo", "run_*", "evaluations.npz"), axes[0, 1]),
        ("A2C", os.path.join(ROOT, "models", "pg", "a2c", "run_*", "evaluations.npz"), axes[1, 0]),
    ]
    for title, pattern, ax in specs:
        paths = sorted(glob.glob(pattern))
        for p in paths[:10]:
            try:
                data = np.load(p, allow_pickle=True)
            except Exception:
                continue
            if "timesteps" not in data or "results" not in data:
                continue
            ts = np.asarray(data["timesteps"]).flatten()
            res = np.asarray(data["results"])
            if res.ndim == 2:
                y = np.mean(res, axis=1)
            else:
                y = res
            ax.plot(ts, y, alpha=0.5, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean eval return")
        ax.grid(True, alpha=0.3)

    # REINFORCE: episode returns from jsonl
    ax = axes[1, 1]
    rf = sorted(glob.glob(os.path.join(ROOT, "models", "pg", "reinforce_run_*_metrics.jsonl")))[:10]
    for p in rf:
        rows = _load_jsonl(p)
        if not rows:
            continue
        ep = [r["episode"] for r in rows]
        ret = [r["return"] for r in rows]
        ax.plot(ep, ret, alpha=0.45, linewidth=1)
    ax.set_title("REINFORCE (episode return)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "01_convergence_and_returns.png"), dpi=150)
    plt.close(fig)


def _series_for_key(rows: List[Dict[str, Any]], key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        if key not in row:
            continue
        xs.append(float(row.get("timestep", row.get("n_calls", 0))))
        ys.append(float(row[key]))
    return xs, ys


def plot_dqn_objective() -> None:
    import matplotlib.pyplot as plt

    paths = sorted(glob.glob(os.path.join(ROOT, "models", "dqn", "run_*", "metrics.jsonl")))[:10]
    if not paths:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for p in paths:
        rows = _load_jsonl(p)
        label = os.path.basename(os.path.dirname(p))
        lx, ly = _series_for_key(rows, "train/loss")
        if ly:
            axes[0].plot(lx, ly, alpha=0.65, label=label)
        rx, ry = _series_for_key(rows, "rollout/ep_rew_mean")
        if ry:
            axes[1].plot(rx, ry, alpha=0.65, label=label)
    axes[0].set_title("DQN train/loss (TD-style objective, when present)")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title("DQN rollout episode reward (training env)")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Ep rew mean")
    axes[1].grid(True, alpha=0.3)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_dqn_objective.png"), dpi=150)
    plt.close(fig)


def plot_pg_entropy() -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, algo, sub in zip(
        axes,
        ("PPO", "A2C"),
        ("ppo", "a2c"),
    ):
        paths = sorted(glob.glob(os.path.join(ROOT, "models", "pg", sub, "run_*", "metrics.jsonl")))[:10]
        for p in paths:
            rows = _load_jsonl(p)
            xs, ys, key = _pick_scalar_row(
                rows,
                ("train/entropy_loss", "train/entropy", "entropy_loss"),
            )
            if ys:
                ax.plot(xs, ys, alpha=0.55, linewidth=1)
        ax.set_title(f"{algo} entropy-related log (SB3 key varies)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Logged value")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Policy gradient training: entropy / entropy_loss curves", fontsize=12)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "03_pg_entropy.png"), dpi=150)
    plt.close(fig)


def plot_reinforce_entropy() -> None:
    import matplotlib.pyplot as plt

    paths = sorted(glob.glob(os.path.join(ROOT, "models", "pg", "reinforce_run_*_metrics.jsonl")))[:10]
    if not paths:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for p in paths:
        rows = _load_jsonl(p)
        if not rows:
            continue
        ep = [r["episode"] for r in rows]
        h = [r.get("mean_entropy", 0.0) for r in rows]
        ax.plot(ep, h, alpha=0.55)
    ax.set_title("REINFORCE: policy entropy (episode mean)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean entropy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "04_reinforce_entropy.png"), dpi=150)
    plt.close(fig)


def plot_mean_eval_bars() -> None:
    import matplotlib.pyplot as plt

    def load_summaries(glob_pat: str) -> List[Optional[float]]:
        vals: List[Optional[float]] = []
        for p in sorted(glob.glob(glob_pat)):
            try:
                with open(p) as f:
                    d = json.load(f)
                vals.append(d.get("mean_eval"))
            except Exception:
                vals.append(None)
        return vals

    dqn = load_summaries(os.path.join(ROOT, "models", "dqn", "run_*", "summary.json"))
    ppo = load_summaries(os.path.join(ROOT, "models", "pg", "ppo", "run_*", "summary.json"))
    a2c = load_summaries(os.path.join(ROOT, "models", "pg", "a2c", "run_*", "summary.json"))
    rf: List[Optional[float]] = []
    for p in sorted(glob.glob(os.path.join(ROOT, "models", "pg", "reinforce_run_*.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
            rf.append(d.get("mean_return_last_50"))
        except Exception:
            rf.append(None)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(10)
    width = 0.2

    def bar(series: List[Optional[float]], offset: float, label: str):
        ys = [float(v) if v is not None else 0.0 for v in series[:10]]
        ax.bar(x + offset, ys, width, label=label, alpha=0.85)

    if dqn:
        bar(dqn, -1.5 * width, "DQN mean_eval")
    if ppo:
        bar(ppo, -0.5 * width, "PPO mean_eval")
    if a2c:
        bar(a2c, 0.5 * width, "A2C mean_eval")
    if rf:
        bar(rf, 1.5 * width, "REINFORCE last-50 mean")

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("Run id (0–9)")
    ax.set_ylabel("Score (see legend units)")
    ax.set_title("Per-run scores from summary.json (train & compare)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "05_run_score_bars.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_eval_convergence()
    plot_dqn_objective()
    plot_pg_entropy()
    plot_reinforce_entropy()
    plot_mean_eval_bars()
    print(f"Wrote figures under: {OUT_DIR}")


if __name__ == "__main__":
    main()
