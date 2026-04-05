"""
Log SB3 logger scalars for summative plots (DQN loss, policy entropy, value loss, etc.).
Each line is one JSON object (variable keys OK).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SummativeMetricsJSONLCallback(BaseCallback):
    """Append one JSON record every `log_every` env steps when logger has values."""

    def __init__(self, jsonl_path: str, log_every: int = 256):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.log_every = max(1, int(log_every))

    def _init_callback(self) -> None:
        d = os.path.dirname(self.jsonl_path)
        if d:
            os.makedirs(d, exist_ok=True)
        open(self.jsonl_path, "w").close()

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every != 0:
            return True
        row: Dict[str, Any] = {
            "timestep": int(self.num_timesteps),
            "n_calls": int(self.n_calls),
        }
        if self.model.logger is not None and self.model.logger.name_to_value:
            for k, v in self.model.logger.name_to_value.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    row[str(k)] = float(v)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return True
