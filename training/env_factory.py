"""
Single source of truth for the summative training environment (production MDP).
"""

from __future__ import annotations

from stable_baselines3.common.monitor import Monitor

from environment.kulima_production_env import KulimaProductionEnv

# Aligned with KulimaProductionEnv default; longer horizon than legacy grid demo.
PRODUCTION_MAX_STEPS = 250


def make_monitored_env(seed: int = 0):
    """SB3 VecEnv factory: Monitor wrapper + deterministic seed at reset."""

    def _init():
        e = KulimaProductionEnv(render_mode=None, max_steps=PRODUCTION_MAX_STEPS)
        e.reset(seed=seed)
        return Monitor(e)

    return _init
