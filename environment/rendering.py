"""
Pygame-based visualization for KulimaFarmMissionEnv.
Supports human window and rgb_array for video / GIF export.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from environment.custom_env import KulimaFarmMissionEnv

import pygame

CELL_PIXELS = 48
_COLORS = {
    0: (34, 139, 34),  # EMPTY grass green
    1: (50, 205, 50),  # healthy field
    2: (178, 34, 34),  # diseased
    3: (30, 144, 255),  # weather tower
    4: (255, 215, 0),  # market
    5: (139, 69, 19),  # extension hub
    6: (40, 40, 40),  # blocked
}

_screen = None
_clock = None
_font: Optional[pygame.font.Font] = None


def _ensure_init():
    global _font
    if pygame.get_init() is False:
        pygame.init()
    if _font is None:
        _font = pygame.font.SysFont("arial", 14)


def close_pygame():
    global _screen, _clock, _font
    if pygame.get_init():
        pygame.quit()
    _screen = None
    _clock = None
    _font = None


def render_grid_pygame(env: "KulimaFarmMissionEnv", mode: str = "human") -> Optional[np.ndarray]:
    _ensure_init()
    grid = env._grid
    h, w = grid.shape
    height_px = h * CELL_PIXELS + 80
    width_px = w * CELL_PIXELS

    global _screen, _clock
    if _screen is None or _screen.get_size() != (width_px, height_px):
        if mode == "human":
            _screen = pygame.display.set_mode((width_px, height_px))
            pygame.display.set_caption("KulimaIQ Mission RL — KulimaFarmMissionEnv")
        else:
            _screen = pygame.Surface((width_px, height_px))
        _clock = pygame.time.Clock()

    assert _screen is not None
    surf = _screen
    surf.fill((20, 60, 20))

    for r in range(h):
        for c in range(w):
            cell = int(grid[r, c])
            color = _COLORS.get(cell, (100, 100, 100))
            rect = pygame.Rect(c * CELL_PIXELS, r * CELL_PIXELS, CELL_PIXELS, CELL_PIXELS)
            pygame.draw.rect(surf, color, rect.inflate(-2, -2))
            pygame.draw.rect(surf, (0, 0, 0), rect, 1)

    ar, ac = env._agent_pos
    cx = ac * CELL_PIXELS + CELL_PIXELS // 2
    cy = ar * CELL_PIXELS + CELL_PIXELS // 2
    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), CELL_PIXELS // 4)
    pygame.draw.circle(surf, (0, 0, 0), (cx, cy), CELL_PIXELS // 4, 2)

    legend_y = h * CELL_PIXELS + 4
    if _font:
        flags = f"D:{int(env._diag_done)} C:{int(env._climate_done)} M:{int(env._market_done)} step:{env._step_count}"
        txt = _font.render(flags, True, (255, 255, 255))
        surf.blit(txt, (4, legend_y))

    if mode == "human":
        pygame.event.pump()
        pygame.display.flip()
        if _clock:
            _clock.tick(env.metadata.get("render_fps", 8))
        return None

    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
