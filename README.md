# KulimaIQ_Summative

**KulimaIQ-inspired reinforcement learning summative:** four algorithms (**DQN**, **REINFORCE**, **PPO**, **A2C**) on a custom **Gymnasium** environment that models a **production-style agricultural pipeline**—edge diagnosis, climate/forecast integration, and market listing—under **stochastic** dynamics (no grid-world abstraction).

> **GitHub:** publish this project as repository **`KulimaIQ_Summative`** (URL: `https://github.com/<your-username>/KulimaIQ_Summative`). After `git clone`, work from that folder.

---

## Video demonstration

**Summative demo (camera on, GUI + objective/rewards explained):**  
[https://youtu.be/7P_V84w7MdE](https://youtu.be/7P_V84w7MdE)

The recording should show **full screen**, **Pygame dashboard**, and **terminal** output (`play.py` / `main.py`), and state the **mission objective**, **reward structure**, and **agent behavior** (see checklist below).

---

## Full written report

A **~10-page** summative document (abstract, methods, filled hyperparameter tables, results, convergence & generalization tables, ethics) lives in:

**[`RL_Summative_Assignment_Report.md`](RL_Summative_Assignment_Report.md)**

Use it as the source for submission PDF/Word; this README is the **project landing page** with the same narrative in shorter form.

---

## Problem and approach

Smallholder-facing stacks must **sequence** expensive steps—on-device inference, **API** calls (weather, listings), notifications, **human escalation**—under **bandwidth, battery, and risk**. Fixed rules break when **noise is correlated** and **rewards are delayed**.

This project treats the controller as an **MDP**: at each step the agent sees a **normalized observation vector**, picks **one** of **ten pipeline actions**, receives a **shaped reward**, and the world **drifts** stochastically (infection, weather regime, markets, sync backlog). Learning uses **Stable-Baselines3** (DQN, PPO, A2C) plus **custom PyTorch REINFORCE**, each with **≥10 hyperparameter configurations** in `training/hyperparams.py`.

---

## Environment: `KulimaProductionEnv`

Primary implementation: [`environment/kulima_production_env.py`](environment/kulima_production_env.py). Visualization: [`environment/rendering_pipeline.py`](environment/rendering_pipeline.py) (Pygame **bar dashboard** + mission checklist).

| | |
|--|--|
| **Observation** | `Box(0, 1, shape=(18,))` — crop vigor, infection/symptoms, soil moisture, forecast confidence, rain pressure, market spread, logistics friction, sync queue, model uncertainty, time remaining, extension load, **three mission flags** (diagnosis / climate / market), weather regime, reward EMA, diagnosis cooldown |
| **Actions** | `Discrete(10)` — monitor, edge diagnosis batch, fetch/merge forecast, seasonal alert, market listing, flush sync queue, throttle inference, extension escalation, field scout notification, early warning |
| **Rewards** | Per-step cost (−0.025); milestones (e.g. diagnosis +5.5, climate +3.5, market up to +5); completion +12; failure/timeout penalties; **edge-case** penalties (cooldown abuse, low-confidence alerts, warnings when risk is low) |
| **Termination** | All missions complete, crop failure (vigor too low), or `max_steps` (default **250**) |
| **Generalization test** | `dynamics_shift > 1` amplifies drift (see `analysis/generalization_eval.py`) |

**Legacy:** [`environment/custom_env.py`](environment/custom_env.py) (`KulimaFarmMissionEnv`, grid-style) is reference only. If you have **old** checkpoints, **retrain**—observation size and action count differ from production env.

**Climate note:** There are **no** separate “sun” or “heat” state variables; weather enters via **rain pressure**, **weather regime** (dry / mixed / wet), **soil moisture**, and **forecast confidence**.

---

## Algorithms (summary)

| Method | Implementation | Highlights |
|--------|----------------|------------|
| **DQN** | SB3 `DQN` + `MlpPolicy` | Replay buffer, target network, ε-greedy decay |
| **REINFORCE** | PyTorch `PolicyNet` in `training/reinforce_training.py` | Monte Carlo returns, per-episode return norm, optional entropy, grad clip |
| **PPO** | SB3 `PPO` | Clipped surrogate, GAE, `ent_coef`, `clip_range` |
| **A2C** | SB3 `A2C` | Small `n_steps` in sweep (fast, noisy updates) |

Training logs: TensorBoard under each run, **`metrics.jsonl`** (SB3), **`reinforce_run_*_metrics.jsonl`** (REINFORCE). Best checkpoints: **`models/*/dqn_best.json`**, **`ppo_best.json`**, **`a2c_best.json`**, **`reinforce_best.json`**.

---

## Results snapshot *(from saved `summary.json` — see report for tables)*

- **Highest mean eval (40k-step SB3 runs):** **DQN run 6** ≈ **96.15**; **A2C** runs **2 / 5 / 8** ≈ **96.0**.
- **Best REINFORCE (last-50-episode mean):** **run 4** ≈ **94.10** (high variance across other rows).
- **PPO pilot sweep** (shorter budget per run): best **run 5** ≈ **81.23** — **re-run PPO** with the same `RL_TIMESTEPS` as DQN/A2C for a fair comparison.
- **Checkpoint vs env:** Many **DQN / A2C / REINFORCE** artifacts were trained with **11-D** policies; **`KulimaProductionEnv` is 18-D**. For **`play.py` on the current env**, use a policy trained on **18-D** (e.g. **PPO** after your latest sweep) until everything is retrained.

**Generalization (example):** Measured **PPO** on 18-D: baseline mean return **~78.8**, under `dynamics_shift=1.25` **~57.9** (~**26.5%** drop). See report §6.2 for full table and scaling notes for other algorithms.

---

## Project layout

```
KulimaIQ_Summative/
├── environment/
│   ├── kulima_production_env.py   # primary MDP
│   ├── custom_env.py              # legacy grid env
│   ├── rendering.py
│   └── rendering_pipeline.py      # Pygame dashboard
├── training/
│   ├── hyperparams.py             # 10 configs × DQN, PPO, A2C, REINFORCE
│   ├── env_factory.py
│   ├── metrics_callback.py
│   ├── dqn_training.py
│   ├── pg_training.py             # --algo ppo | a2c
│   └── reinforce_training.py
├── analysis/
│   ├── plot_summative.py
│   ├── export_hyperparameter_tables.py
│   └── generalization_eval.py
├── models/                        # checkpoints + summary JSON (gitignored patterns may apply)
├── demos/
├── diagrams/
├── RL_Summative_Assignment_Report.md
├── main.py
├── play.py                        # rubric demo entry (= main CLI)
├── api_server.py                  # optional FastAPI JSON API
├── requirements.txt
└── README.md
```

---

## Install

```bash
cd KulimaIQ_Summative
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Training (≥10 runs per algorithm)

```bash
export RL_TIMESTEPS=200000   # default 40000 if unset
python training/dqn_training.py
python training/reinforce_training.py
python training/pg_training.py --algo ppo
python training/pg_training.py --algo a2c
```

---

## Run trained policy (GUI)

```bash
python play.py --algo ppo --episodes 5 --render human
# or: python main.py --algo ppo --episodes 5 --render human
```

Use **`--algo ppo`** (or any model **trained on 18-D** `KulimaProductionEnv`) if **`dqn`** fails with an observation-shape error on legacy checkpoints.

---

## Analysis & rubric artifacts

```bash
python analysis/plot_summative.py
python analysis/export_hyperparameter_tables.py
python analysis/generalization_eval.py --algo ppo --episodes 30
```

Outputs: **`results/figures/`**, **`results/hyperparameter_tables.md`**.

---

## JSON API *(integration sketch)*

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8765
```

Endpoints: `POST /reset?seed=…`, `GET /state`, `POST /step` with `{"action": 0–9}`. State: `KulimaProductionEnv.serialize_state()`.

---

## Random policy GIF

```bash
python demos/static_random_demo.py
```

Writes `demos/output/random_agent_kulima_pipeline.gif`.

---

## Video / submission checklist

- [ ] Link live: [Demonstration on YouTube](https://youtu.be/7P_V84w7MdE)
- [ ] Full screen, camera on if required
- [ ] State **objective** (diagnosis → climate → market under uncertainty)
- [ ] Explain **rewards** (milestones, penalties, completion/failure)
- [ ] Show **Pygame** + **terminal** verbose
- [ ] Interpret **policy** vs random / naive sequencing

---

## References *(short)*

- Mnih *et al.* (2015). Human-level control through deep RL. *Nature*. [DOI](https://doi.org/10.1038/nature14236)
- Schulman *et al.* (2017). PPO. [arXiv](https://arxiv.org/abs/1707.06347)
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) · [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)

---

## License

Educational use — capstone / summative assignment.
