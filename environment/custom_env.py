"""
KulimaFarmMissionEnv — mission-based Gymnasium environment aligned with the KulimaIQ capstone:
smallholder digital agriculture: disease diagnosis, climate advisory, and market linkage.

The agent navigates a grid representing a farm landscape with fields, a weather/climate node,
a market, and an extension hub. Success requires completing three mission objectives in a
resource-aware order (climate and diagnosis before optimal market reward).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

# Cell types in the static map
EMPTY = 0
FIELD_HEALTHY = 1
FIELD_DISEASED = 2  # requires USE_KULIMAIQ_DIAGNOSE while standing on cell
WEATHER_TOWER = 3  # KulimaIQ climate / advisory integration metaphor
MARKET = 4
EXTENSION_HUB = 5
BLOCKED = 6


class KulimaFarmMissionEnv(gym.Env):
    """
    Custom environment: exhaustive discrete actions relevant to KulimaIQ mission metaphor.

    Action space (Discrete(9)):
        0: WAIT — no movement; time passes (small penalty).
        1-4: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
        5: USE_KULIMAIQ_DIAGNOSE — CNN/ML disease check (valid only on FIELD_DISEASED).
        6: FETCH_CLIMATE_ADVISORY — pull localized weather/climate advisory (valid on WEATHER_TOWER).
        7: POST_MARKET_LISTING — list produce / market linkage (valid on MARKET).
        8: CONSULT_EXTENSION — optional: small bonus at EXTENSION_HUB (extension officer metaphor).
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 8}

    # Action indices (document for students / grader)
    ACTION_WAIT = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    ACTION_DIAGNOSE = 5
    ACTION_CLIMATE = 6
    ACTION_MARKET = 7
    ACTION_EXTENSION = 8

    ACTION_NAMES = [
        "WAIT",
        "MOVE_UP",
        "MOVE_DOWN",
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "USE_KULIMAIQ_DIAGNOSE",
        "FETCH_CLIMATE_ADVISORY",
        "POST_MARKET_LISTING",
        "CONSULT_EXTENSION",
    ]

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(9)

        # Observation: normalized position, mission flags, normalized distances, time fraction
        self._obs_dim = 11
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32
        )

        self._grid: np.ndarray
        self._agent_pos: Tuple[int, int]
        self._step_count: int
        self._diag_done: bool
        self._climate_done: bool
        self._market_done: bool
        self._extension_consult_count: int

        self._build_default_map()

    def _build_default_map(self) -> None:
        """9x9 grid: non-generic layout for Byumba-style mission (fields, tower, market, hub)."""
        h, w = 9, 9
        self._grid = np.zeros((h, w), dtype=np.int32)
        # Borders as low obstacles (optional inner paths)
        self._grid[0, :] = BLOCKED
        self._grid[-1, :] = BLOCKED
        self._grid[:, 0] = BLOCKED
        self._grid[:, -1] = BLOCKED
        # Inner walkable region
        inner = slice(1, 8)
        self._grid[inner, inner] = EMPTY
        # Extension hub (start) — north-central inner
        self._grid[2, 4] = EXTENSION_HUB
        # Diseased field — south-west area
        self._grid[5, 2] = FIELD_DISEASED
        self._grid[5, 3] = FIELD_HEALTHY
        self._grid[6, 2] = FIELD_HEALTHY
        # Weather / climate advisory point — east
        self._grid[3, 6] = WEATHER_TOWER
        # Market — south-east
        self._grid[6, 6] = MARKET
        # Extra healthy fields
        self._grid[4, 4] = FIELD_HEALTHY
        self._grid[3, 2] = EMPTY

    def _find_cells(self, cell_type: int) -> List[Tuple[int, int]]:
        ys, xs = np.where(self._grid == cell_type)
        return list(zip(ys.tolist(), xs.tolist()))

    def _start_position(self) -> Tuple[int, int]:
        hubs = self._find_cells(EXTENSION_HUB)
        if not hubs:
            return (4, 4)
        return hubs[0]

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _nearest_distance(self, cell_type: int) -> float:
        targets = self._find_cells(cell_type)
        if not targets:
            return 0.0
        d = min(self._manhattan(self._agent_pos, t) for t in targets)
        max_d = self._grid.shape[0] + self._grid.shape[1]
        return d / max_d

    def _get_obs(self) -> np.ndarray:
        h, w = self._grid.shape
        ar, ac = self._agent_pos
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[0] = ar / (h - 1)
        obs[1] = ac / (w - 1)
        obs[2] = float(self._diag_done)
        obs[3] = float(self._climate_done)
        obs[4] = float(self._market_done)
        obs[5] = self._nearest_distance(FIELD_DISEASED)
        obs[6] = self._nearest_distance(WEATHER_TOWER)
        obs[7] = self._nearest_distance(MARKET)
        obs[8] = 1.0 - min(1.0, self._step_count / self.max_steps)
        # Local cell type one-hot style summary (what is under agent)
        cell = self._grid[ar, ac]
        obs[9] = float(cell == FIELD_DISEASED)
        obs[10] = float(cell in (WEATHER_TOWER, MARKET, EXTENSION_HUB))
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._agent_pos = self._start_position()
        self._step_count = 0
        self._diag_done = False
        self._climate_done = False
        self._market_done = False
        self._extension_consult_count = 0

        return self._get_obs(), {}

    def _move(self, dr: int, dc: int) -> Tuple[float, bool]:
        ar, ac = self._agent_pos
        nr, nc = ar + dr, ac + dc
        h, w = self._grid.shape
        if not (0 <= nr < h and 0 <= nc < w):
            return -0.3, False
        cell = self._grid[nr, nc]
        if cell == BLOCKED:
            return -0.5, False
        self._agent_pos = (nr, nc)
        return -0.02, True  # small step cost bundled in caller

    def _mission_complete(self) -> bool:
        return self._diag_done and self._climate_done and self._market_done

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = -0.01
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        ar, ac = self._agent_pos
        cell = self._grid[ar, ac]

        if action == self.ACTION_WAIT:
            pass
        elif action == self.ACTION_UP:
            r, ok = self._move(-1, 0)
            reward += r
            if not ok and r < -0.2:
                info["blocked"] = True
        elif action == self.ACTION_DOWN:
            r, ok = self._move(1, 0)
            reward += r
            if not ok and r < -0.2:
                info["blocked"] = True
        elif action == self.ACTION_LEFT:
            r, ok = self._move(0, -1)
            reward += r
            if not ok and r < -0.2:
                info["blocked"] = True
        elif action == self.ACTION_RIGHT:
            r, ok = self._move(0, 1)
            reward += r
            if not ok and r < -0.2:
                info["blocked"] = True
        elif action == self.ACTION_DIAGNOSE:
            if cell == FIELD_DISEASED and not self._diag_done:
                self._diag_done = True
                reward += 6.0
                info["event"] = "diagnosis_success"
            else:
                reward -= 0.15
                info["invalid_action"] = "diagnose"
        elif action == self.ACTION_CLIMATE:
            if cell == WEATHER_TOWER and not self._climate_done:
                self._climate_done = True
                reward += 4.0
                info["event"] = "climate_advisory"
            else:
                reward -= 0.15
                info["invalid_action"] = "climate"
        elif action == self.ACTION_MARKET:
            if cell == MARKET and not self._market_done:
                self._market_done = True
                if self._climate_done:
                    reward += 5.0
                else:
                    reward += 1.5
                info["event"] = "market_listing"
            else:
                reward -= 0.15
                info["invalid_action"] = "market"
        elif action == self.ACTION_EXTENSION:
            if cell == EXTENSION_HUB:
                self._extension_consult_count += 1
                reward += 0.5
                info["event"] = "extension_consult"
            else:
                reward -= 0.1
                info["invalid_action"] = "extension"
        else:
            raise ValueError(f"Unknown action {action}")

        self._step_count += 1

        if self._mission_complete():
            reward += 12.0
            terminated = True
            info["mission_success"] = True

        if self._step_count >= self.max_steps:
            truncated = True
            if not self._mission_complete():
                reward -= 2.0
                info["mission_timeout"] = True

        info.update(
            {
                "diag_done": self._diag_done,
                "climate_done": self._climate_done,
                "market_done": self._market_done,
            }
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        try:
            from .rendering import render_grid_pygame
        except ImportError:
            from environment.rendering import render_grid_pygame

        return render_grid_pygame(self, mode=self.render_mode)

    def close(self) -> None:
        try:
            from .rendering import close_pygame
        except ImportError:
            from environment.rendering import close_pygame

        close_pygame()
