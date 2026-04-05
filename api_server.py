#!/usr/bin/env python3
"""
Minimal JSON API over KulimaProductionEnv for frontend / product integration demos.

Run (from project root, after `pip install -r requirements.txt`):

    uvicorn api_server:app --reload --host 0.0.0.0 --port 8765

Endpoints:
    POST /reset?seed=    — optional query: seed (int)
    GET  /state          — serialized observation + mission flags
    POST /step           — body: {"action": int} (0–9)
    GET  /action_names   — human-readable pipeline actions
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from environment.kulima_production_env import KulimaProductionEnv
from training.env_factory import PRODUCTION_MAX_STEPS

app = FastAPI(title="KulimaIQ RL Env API", version="1.0.0")

_env: Optional[KulimaProductionEnv] = None


def get_env() -> KulimaProductionEnv:
    global _env
    if _env is None:
        _env = KulimaProductionEnv(render_mode=None, max_steps=PRODUCTION_MAX_STEPS)
        _env.reset()
    return _env


class StepBody(BaseModel):
    action: int = Field(..., ge=0, le=9)


@app.post("/reset")
def reset(seed: Optional[int] = Query(default=None)) -> Dict[str, Any]:
    e = get_env()
    obs, info = e.reset(seed=seed)
    out = e.serialize_state()
    out["info"] = {k: v for k, v in info.items() if k != "action_names"}
    return out


@app.get("/state")
def state() -> Dict[str, Any]:
    e = get_env()
    return e.serialize_state()


@app.post("/step")
def step(body: StepBody) -> Dict[str, Any]:
    e = get_env()
    try:
        obs, reward, term, trunc, info = e.step(body.action)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return {
        "observation": obs.tolist(),
        "reward": reward,
        "terminated": term,
        "truncated": trunc,
        "info": {k: v for k, v in info.items() if k != "action_names"},
        "serialized": e.serialize_state(),
    }


@app.get("/action_names")
def action_names() -> Dict[str, Any]:
    return {"actions": KulimaProductionEnv.ACTION_NAMES}
