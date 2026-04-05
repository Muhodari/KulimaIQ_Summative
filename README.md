# sage_muhodari_rl_summative

**Mission-aligned RL summative:** compares **DQN** (value-based), **REINFORCE** (vanilla policy gradient), **PPO**, and **A2C** (actor–critic) on a **custom Gymnasium** environment inspired by **KulimaIQ** (edge diagnosis, climate advisory, market linkage for smallholders in Eastern Africa).

> **GitHub:** create a repository named `sage_muhodari_rl_summative` (or `student_name_rl_summative` per your course) and push this folder.

## Primary environment (production-style MDP)

Training and `main.py` / `play.py` use **`KulimaProductionEnv`** (`environment/kulima_production_env.py`): **continuous** agronomic and operational state (crop vigor, infection pressure, forecast quality, sync backlog, market spread, etc.), **stochastic** transitions, and **Discrete(10)** pipeline actions (diagnosis batch, forecast fetch, alerts, listing, sync flush, extension escalation, …). This is intended to mirror integration realism more closely than a spatial grid.

**Legacy reference:** `KulimaFarmMissionEnv` in `environment/custom_env.py` remains for comparison / older diagrams; **retrain all algorithms** if you previously used checkpoints from the grid env (action count and observation size differ).

| Item | `KulimaProductionEnv` |
|------|------------------------|
| **Action space** | `Discrete(10)` — see `ACTION_NAMES` in `kulima_production_env.py` |
| **Observation** | `Box(0,1, shape=(18,))` — normalized signals + mission bits + regime hints |
| **Rewards** | Step cost, milestone bonuses (diagnosis / climate / market), completion bonus, failure penalties |
| **Termination** | Mission complete, crop failure, or `max_steps` (default 250) |

Visualization: **`environment/rendering_pipeline.py`** — Pygame dashboard (bars + mission checklist), not a tile grid.

## Project layout

```
project_root/
├── environment/
│   ├── kulima_production_env.py  # primary training env
│   ├── custom_env.py             # legacy grid mission (reference)
│   ├── rendering.py
│   └── rendering_pipeline.py
├── training/
│   ├── env_factory.py
│   ├── metrics_callback.py
│   ├── hyperparams.py
│   ├── dqn_training.py
│   ├── pg_training.py
│   └── reinforce_training.py
├── analysis/
│   ├── plot_summative.py
│   ├── export_hyperparameter_tables.py
│   └── generalization_eval.py
├── models/
├── demos/
├── diagrams/
├── main.py
├── play.py              # rubric demo entry (same CLI as main.py)
├── api_server.py        # optional FastAPI JSON API
├── requirements.txt
└── README.md
```

## Install

```bash
cd sage_muhodari_rl_summative
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training (≥10 hyperparameter runs per algorithm)

Configurations live in `training/hyperparams.py`. Longer training:

```bash
export RL_TIMESTEPS=200000   # default 40000 if unset
```

```bash
python training/dqn_training.py
python training/reinforce_training.py
python training/pg_training.py --algo ppo
python training/pg_training.py --algo a2c
```

**Artifacts:** SB3 runs log **`metrics.jsonl`** (logger scalars) beside TensorBoard. REINFORCE logs per-episode **`reinforce_run_*_metrics.jsonl`**.

## Run trained policy (GUI)

```bash
python play.py --algo dqn --episodes 5 --render human
# equivalent:
python main.py --algo dqn --episodes 5 --render human
```

## Analysis & tables (rubric)

After training:

```bash
python analysis/plot_summative.py
python analysis/export_hyperparameter_tables.py
python analysis/generalization_eval.py --algo dqn --episodes 30
```

Figures → `results/figures/`. Hyperparameter markdown → `results/hyperparameter_tables.md`.

## JSON API (product integration sketch)

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8765
```

`POST /reset?seed=…`, `GET /state`, `POST /step` with `{"action": 0–9}`. Environment state can also be read via `KulimaProductionEnv.serialize_state()` for custom backends.

## Random policy demo (GIF)

```bash
python demos/static_random_demo.py
```

Writes `demos/output/random_agent_kulima_pipeline.gif`.

## Video demonstration checklist (assignment)

- Full-screen capture, webcam on if required.
- State the **objective** (complete diagnosis → climate → market pipeline under uncertainty).
- Explain **reward** (milestones, penalties, completion).
- Show **GUI** (Pygame) **and** terminal logs (`play.py` / `main.py`).
- Interpret **agent behavior** (sequence of actions vs. random baseline).

## Course naming

Rename the root folder / GitHub repo to match your instructor’s pattern, e.g. `firstname_lastname_rl_summative`.

## License

Educational use — capstone / summative assignment.
