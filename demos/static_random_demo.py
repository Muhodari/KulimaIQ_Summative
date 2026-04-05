#!/usr/bin/env python3
"""
Static demonstration: random agent in KulimaProductionEnv (no RL training).
Saves an animated GIF under demos/output/ for submission / report embedding.

Run from project root:
    python demos/static_random_demo.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

import imageio.v2 as imageio

from environment.kulima_production_env import KulimaProductionEnv


def main():
    out_dir = os.path.join(ROOT, "demos", "output")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, "random_agent_kulima_pipeline.gif")

    env = KulimaProductionEnv(render_mode="rgb_array", max_steps=120)
    obs, _ = env.reset(seed=42)

    frames = []
    rng = np.random.default_rng(123)

    for _ in range(100):
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        action = int(rng.integers(0, env.action_space.n))
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 10_000)))

    env.close()

    if frames:
        imageio.mimsave(gif_path, frames, duration=0.12, loop=0)
        print(f"Saved GIF: {gif_path} ({len(frames)} frames)")
    else:
        print("No frames captured.")


if __name__ == "__main__":
    main()
