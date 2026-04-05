"""
Pygame dashboard for KulimaProductionEnv — continuous signals + mission pipeline (not a grid).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from .kulima_production_env import KulimaProductionEnv

_screen = None
_clock = None
_font = None
_small_font = None


def _ensure_pygame():
    import pygame

    return pygame


def close_pipeline_render() -> None:
    global _screen, _clock, _font, _small_font
    if _screen is not None:
        try:
            import pygame

            pygame.quit()
        except Exception:
            pass
    _screen = _clock = _font = _small_font = None


def _bar(
    surf,
    pygame,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    pygame.draw.rect(surf, (40, 44, 52), (x, y, w, h), border_radius=4)
    fill_w = int(w * max(0, min(1, value)))
    pygame.draw.rect(surf, color, (x, y, fill_w, h), border_radius=4)
    pygame.draw.rect(surf, (200, 200, 210), (x, y, w, h), 1, border_radius=4)
    if _small_font:
        t = _small_font.render(f"{label}: {value:.2f}", True, (230, 232, 240))
        surf.blit(t, (x + 4, y + 2))


def render_production_dashboard(env: "KulimaProductionEnv", mode: str) -> Optional[Any]:
    global _screen, _clock, _font, _small_font
    pygame = _ensure_pygame()
    W, H = 920, 520

    if mode == "human":
        if _screen is None:
            pygame.init()
            pygame.display.set_caption("KulimaIQ Production Pipeline — RL Viz")
            _screen = pygame.display.set_mode((W, H))
            _clock = pygame.time.Clock()
            _font = pygame.font.SysFont("menlo", 22)
            _small_font = pygame.font.SysFont("menlo", 15)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

        surf = pygame.Surface((W, H))
        surf.fill((24, 26, 32))

        title = _font.render("KulimaIQ operational control (continuous state)", True, (180, 220, 180))
        surf.blit(title, (20, 12))

        y0 = 48
        row_h = 28
        gap = 6
        x, bw = 20, 420

        _bar(surf, pygame, x, y0, bw, row_h, env._crop_vigor, "Crop vigor", (80, 180, 120))
        _bar(surf, pygame, x, y0 + row_h + gap, bw, row_h, env._latent_infection, "Latent infection", (200, 100, 100))
        _bar(surf, pygame, x, y0 + 2 * (row_h + gap), bw, row_h, env._symptom_index, "Symptom visibility", (220, 140, 80))
        _bar(surf, pygame, x, y0 + 3 * (row_h + gap), bw, row_h, env._forecast_confidence, "Forecast confidence", (100, 160, 220))
        _bar(surf, pygame, x, y0 + 4 * (row_h + gap), bw, row_h, env._sync_queue, "Offline sync queue", (180, 120, 200))
        _bar(surf, pygame, x, y0 + 5 * (row_h + gap), bw, row_h, env._market_spread, "Market opportunity", (200, 200, 100))

        x2 = 460
        _bar(surf, pygame, x2, y0, bw, row_h, env._rain_pressure, "Rain pressure", (90, 140, 200))
        _bar(surf, pygame, x2, y0 + row_h + gap, bw, row_h, env._soil_moisture, "Soil moisture index", (70, 130, 190))
        _bar(surf, pygame, x2, y0 + 2 * (row_h + gap), bw, row_h, env._logistics_friction, "Logistics friction", (160, 100, 100))
        _bar(surf, pygame, x2, y0 + 3 * (row_h + gap), bw, row_h, env._model_uncertainty, "Model uncertainty", (150, 150, 160))
        _bar(surf, pygame, x2, y0 + 4 * (row_h + gap), bw, row_h, env._extension_load, "Extension load", (200, 160, 140))

        # Mission checklist
        my = y0 + 6 * (row_h + gap) + 10
        def chk(label: str, done: bool, ix: int):
            col = (100, 200, 120) if done else (90, 90, 95)
            t = _font.render(f"[{'x' if done else ' '}] {label}", True, col)
            surf.blit(t, (20 + ix * 280, my))

        chk("Diagnosis pipeline", env._m_diag, 0)
        chk("Climate integrated", env._m_climate, 1)
        chk("Market listing", env._m_market, 2)

        st = _small_font.render(
            f"Step {env._step_count}/{env.max_steps}  |  regime={env._weather_regime} (0=dry,1=mixed,2=wet)  |  dynamics_shift={env.dynamics_shift:.2f}",
            True,
            (160, 165, 175),
        )
        surf.blit(st, (20, H - 52))

        leg = _small_font.render(
            "Actions: 0=monitor 1=diagnosis 2=forecast 3=alert 4=listing 5=sync 6=throttle 7=extension 8=scout 9=warning",
            True,
            (130, 135, 145),
        )
        surf.blit(leg, (20, H - 30))

        _screen.blit(surf, (0, 0))
        pygame.display.flip()
        _clock.tick(env.metadata.get("render_fps", 12))
        return None

    if mode == "rgb_array":
        # Headless buffer for recording
        if _screen is None:
            pygame.init()
        surf = pygame.Surface((W, H))
        surf.fill((24, 26, 32))
        # Minimal redraw without display (reuse bars via temp assignment — duplicate quick path)
        import numpy as np

        # Simpler: call human path once without flip — for rgb_array use offscreen
        osurf = pygame.Surface((W, H))
        osurf.fill((24, 26, 32))
        if _font is None:
            _font = pygame.font.SysFont("menlo", 22)
            _small_font = pygame.font.SysFont("menlo", 15)
        y0 = 48
        row_h, gap, x, bw = 28, 6, 20, 420
        _bar(osurf, pygame, x, y0, bw, row_h, env._crop_vigor, "Crop vigor", (80, 180, 120))
        _bar(osurf, pygame, x, y0 + row_h + gap, bw, row_h, env._latent_infection, "Latent infection", (200, 100, 100))
        _bar(osurf, pygame, x, y0 + 2 * (row_h + gap), bw, row_h, env._symptom_index, "Symptom", (220, 140, 80))
        x2 = 460
        _bar(osurf, pygame, x2, y0, bw, row_h, env._rain_pressure, "Rain", (90, 140, 200))
        t = _font.render("KulimaIQ pipeline (rgb_array)", True, (200, 200, 200))
        osurf.blit(t, (20, 12))
        arr = pygame.surfarray.array3d(osurf)
        return np.transpose(arr, (1, 0, 2))

    return None
