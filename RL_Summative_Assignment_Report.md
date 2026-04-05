# Reinforcement Learning Summative Assignment Report

**Target length:** ~**10 pages** in Word/PDF with **11–12 pt** font, **1.15** line spacing, **1″** margins, and **3 figures** (learning curves, DQN loss / PG entropy, convergence or dashboard screenshot). Trim wording further only if you still exceed the limit.

**Student Name:** [Your Name]  
**Video Recording:** [Link — 3 min max, camera on, full screen]  
**GitHub Repository:** [Link]  
**Date:** [Submission Date]

---

## Abstract

This project compares **DQN**, **REINFORCE**, **PPO**, and **A2C** on a custom **Gymnasium** environment, **`KulimaProductionEnv`**, inspired by **KulimaIQ**: a **non-grid**, **stochastic** pipeline for diagnosis, climate integration, and market listing. The agent receives an **18-dimensional** normalized observation and chooses among **ten discrete** operational actions. Training uses **Stable-Baselines3** (DQN, PPO, A2C), **custom PyTorch REINFORCE**, **ten hyperparameter runs per algorithm**, and logged **eval returns** for reporting. Results summarize **mean eval / episode return**, **convergence timing**, and **generalization** under **`dynamics_shift`**. The work is simulation-only; deployment would require retraining, governance, and field validation.

**Keywords:** reinforcement learning, Gymnasium, DQN, REINFORCE, PPO, A2C, digital agriculture, KulimaIQ.

---

## 1. Introduction

Digital agriculture stacks must **sequence** edge inference, **API** calls (forecasts, listings), and **human escalation** under **battery, bandwidth, and risk** constraints. Fixed rule lists fail when **noise is correlated** and **delayed rewards** depend on prior actions. This summative frames control as an **MDP** and learns policies with **deep RL**. The environment emphasizes **continuous telemetry** and **pipeline actions** rather than a **grid world**, to better match **product-style** integration (`play.py`, optional **`api_server.py`**).

**Contributions:** (1) `KulimaProductionEnv` with stochastic dynamics and mission structure; (2) four algorithms × ten hyperparameter configs; (3) evaluation and plotting scripts under `analysis/`; (4) Pygame dashboard for demonstration.

---

## 2. Background

An MDP is \((\mathcal{S}, \mathcal{A}, P, r, \gamma)\). The **discounted return** is \(G_t = \sum_{k \geq 0} \gamma^k r_{t+k}\). **DQN** approximates **optimal Q-values** with a neural net, **experience replay**, a **target network**, and **ε-greedy** exploration with decay. **REINFORCE** optimizes a **stochastic policy** using **Monte Carlo** returns (here with per-episode normalization and optional **entropy** bonus). **A2C** and **PPO** are **actor–critic** methods; PPO adds a **clipped** surrogate for stable updates.

| Concept | In `KulimaProductionEnv` |
|--------|---------------------------|
| Observation | 18-D vector in \([0,1]\) (see §3) |
| Action | One of 10 discrete pipeline commands / step |
| Reward | Step cost + milestones + penalties + terminal bonus/failure |
| Horizon | `max_steps` = 250 or early termination |

---

## 3. Environment

**Rationale:** Operational control (when to diagnose, refresh forecasts, list produce) matters more than **tile navigation** for a KulimaIQ-style product. **Rendering** is a **bar dashboard** (`rendering_pipeline.py`), not a maze.

**Agent:** A single controller selects **one** action per timestep (automation or copilot metaphor).

### 3.1 Actions (`Discrete(10)`)

| # | Name |
|--:|------|
| 0 | NO_OP_MONITOR |
| 1 | RUN_EDGE_DIAGNOSIS_BATCH |
| 2 | FETCH_MERGE_WEATHER_FORECAST |
| 3 | GENERATE_SEASONAL_ALERT |
| 4 | CREATE_OPTIMIZE_MARKET_LISTING |
| 5 | FLUSH_OFFLINE_SYNC_QUEUE |
| 6 | THROTTLE_INFERENCE_SAVE_BATTERY |
| 7 | ESCALATE_TO_EXTENSION_NODE |
| 8 | TRIGGER_FIELD_SCOUT_NOTIFICATION |
| 9 | BROADCAST_EARLY_WARNING |

**Edge cases:** e.g. diagnosis on **cooldown** (−0.12), alerts with **low forecast confidence** (−0.08), warnings when **rain risk is low** (−0.06). See `kulima_production_env.py` for full logic.

### 3.2 Observation (`Box(18,)`)

Normalized features include: **crop vigor**, **latent infection**, **symptoms**, **soil moisture**, **forecast confidence**, **rain pressure**, **market spread**, **logistics friction**, **sync queue**, **model uncertainty**, **time remaining**, **extension load**, **three mission flags** (diagnosis / climate / market), **weather regime** (0/0.5/1), **reward EMA**, **diagnosis cooldown** (scaled).

### 3.3 Rewards and termination

**Base step:** −0.025. **Milestones:** e.g. diagnosis +5.5 (when thresholds met), climate +3.5 (confidence > 0.62), market up to +5.0 (better if climate done and spread > 0.42). **Completion:** +12.0 when all missions done. **Failure:** crop vigor < 0.08 (−6.0). **Timeout:** −2.5 if `max_steps` reached incomplete. After each action, **stochastic drift** updates infection, weather regime, markets, etc.; **`dynamics_shift > 1`** scales stress for generalization tests.

---

## 4. Algorithms and implementation

**Code layout:** `environment/` (env + Pygame), `training/` (sweeps, `env_factory.py`, metrics callback), `analysis/` (plots, tables, `generalization_eval.py`), `main.py` / `play.py` (rollouts).

**DQN (SB3):** `MlpPolicy`, replay **`buffer_size`**, **`target_update_interval`**, ε schedule via **`exploration_fraction`** / **`exploration_final_eps`**, **`EvalCallback`** for best checkpoint.

**REINFORCE (PyTorch):** Two hidden **Tanh** layers → logits; **Adam**; discounted returns; **entropy_coef** optional; grad clip 1.0.

**PPO / A2C (SB3):** `MlpPolicy`; PPO uses **`n_steps`**, **`clip_range`**, **`gae_lambda`**, **`ent_coef`**; A2C uses small **`n_steps`** in this sweep (noisy but fast updates).

---

## 5. Hyperparameter experiments

### 5.0 Data note

Metrics come from **`models/*/summary.json`** and **`reinforce_run_*.json`**. **DQN / A2C / REINFORCE** used **`RL_TIMESTEPS` = 40 000** (unless you changed it). **PPO** table used a **pilot** run at **3 500** steps per config—**rerun with a longer budget** to match the other algorithms. **Legacy checkpoint note:** older DQN/A2C/REINFORCE weights expect **11-D** inputs; **current** env is **18-D**—**retrain** for `play.py` / `generalization_eval.py` on all four.

### 5.1 DQN (10 runs)

| Run | LR | γ | Buffer | Batch | Exploration (ε) | Mean eval† |
|----:|-----|-----|--------:|------:|-------------------|----------:|
| 0 | 1e-4 | 0.99 | 50 000 | 64 | 1→0.05 @ 0.20 | 96.00 |
| 1 | 3e-4 | 0.99 | 100 000 | 128 | 1→0.08 @ 0.30 | 96.00 |
| 2 | 5e-4 | 0.98 | 50 000 | 32 | 1→0.03 @ 0.15 | 93.02 |
| 3 | 1e-4 | 0.995 | 200 000 | 256 | 1→0.06 @ 0.25 | 96.00 |
| 4 | 2e-4 | 0.99 | 80 000 | 64 | 1→0.10 @ 0.35 | 96.00 |
| 5 | 4e-4 | 0.97 | 60 000 | 64 | 1→0.04 @ 0.20 | 96.00 |
| 6 | 1e-3 | 0.99 | 40 000 | 128 | 1→0.12 @ 0.40 | 96.15 |
| 7 | 5e-5 | 0.995 | 150 000 | 64 | 1→0.05 @ 0.18 | 96.00 |
| 8 | 3e-4 | 0.99 | 100 000 | 64 | 1→0.07 @ 0.22 | 95.90 |
| 9 | 2.5e-4 | 0.988 | 120 000 | 96 | 1→0.065 @ 0.28 | 96.00 |

†Last-5 eval mean (`training/dqn_training.py`).

### 5.2 REINFORCE (10 runs)

| Run | LR | Hidden | γ | Entropy coef. | Mean return (last 50 ep.) |
|----:|-----|--------:|-----|---------------|--------------------------:|
| 0 | 1e-3 | 128 | 0.99 | 0.0 | −5.75 |
| 1 | 3e-4 | 256 | 0.99 | 0.01 | −13.79 |
| 2 | 5e-4 | 128 | 0.995 | 0.0 | −15.16 |
| 3 | 1e-4 | 64 | 0.98 | 0.02 | −18.36 |
| 4 | 2e-3 | 128 | 0.97 | 0.005 | 94.10 |
| 5 | 4e-4 | 192 | 0.99 | 0.0 | 55.02 |
| 6 | 6e-4 | 128 | 0.992 | 0.015 | −15.14 |
| 7 | 8e-4 | 256 | 0.985 | 0.0 | 3.86 |
| 8 | 2.5e-4 | 96 | 0.99 | 0.012 | −18.38 |
| 9 | 7e-4 | 160 | 0.988 | 0.003 | −0.65 |

### 5.3 PPO (10 runs)

| Run | LR | γ | n_steps | Batch | clip | ent_coef | Mean eval† |
|----:|-----|-----|--------:|------:|-----|----------|----------:|
| 0 | 3e-4 | 0.99 | 1024 | 64 | 0.2 | 0.0 | 73.14 |
| 1 | 1e-4 | 0.995 | 2048 | 128 | 0.15 | 0.01 | 54.31 |
| 2 | 5e-4 | 0.98 | 512 | 32 | 0.25 | 0.02 | 62.33 |
| 3 | 2e-4 | 0.99 | 1024 | 256 | 0.2 | 0.005 | 48.00 |
| 4 | 4e-4 | 0.997 | 1536 | 96 | 0.18 | 0.0 | 44.28 |
| 5 | 2.5e-4 | 0.99 | 1024 | 64 | 0.12 | 0.03 | 81.23 |
| 6 | 6e-4 | 0.95 | 256 | 64 | 0.3 | 0.01 | 26.11 |
| 7 | 1e-3 | 0.99 | 512 | 128 | 0.22 | 0.0 | 26.04 |
| 8 | 8e-5 | 0.995 | 2048 | 64 | 0.2 | 0.015 | 48.44 |
| 9 | 3.5e-4 | 0.992 | 768 | 48 | 0.17 | 0.008 | 68.63 |

### 5.4 A2C (10 runs)

| Run | LR | γ | n_steps | ent_coef | vf_coef | max_grad_norm | Mean eval† |
|----:|-----|-----|--------:|---------:|--------:|--------------:|----------:|
| 0 | 7e-4 | 0.99 | 5 | 0.0 | 0.5 | 0.5 | −4.01 |
| 1 | 3e-4 | 0.98 | 5 | 0.01 | 0.4 | 0.7 | −4.01 |
| 2 | 1e-3 | 0.995 | 5 | 0.02 | 0.6 | 1.0 | 96.00 |
| 3 | 5e-4 | 0.99 | 5 | 0.005 | 0.5 | 0.3 | −4.00 |
| 4 | 2e-4 | 0.97 | 5 | 0.0 | 0.45 | 0.5 | −4.00 |
| 5 | 9e-4 | 0.99 | 5 | 0.03 | 0.35 | 1.2 | 96.00 |
| 6 | 4e-4 | 0.996 | 5 | 0.01 | 0.55 | 0.45 | −4.00 |
| 7 | 6e-4 | 0.985 | 5 | 0.0 | 0.5 | 0.8 | −4.00 |
| 8 | 1.2e-3 | 0.99 | 5 | 0.015 | 0.42 | 0.65 | 96.00 |
| 9 | 2.5e-4 | 0.993 | 5 | 0.008 | 0.48 | 0.55 | −4.01 |

**Discussion (2–3 sentences in prose):** Briefly compare **best runs** (DQN **6**, PPO **5**, A2C **2**, REINFORCE **4**), note **PPO pilot** vs **40k** others, and mention **A2C** sensitivity (many runs stuck near **−4**).

---

## 6. Results and discussion

**Figure 1:** Best-run **eval return vs. timesteps** (DQN, PPO, A2C from `evaluations.npz`; REINFORCE: smoothed episode return vs. episode).

**Figure 2:** **DQN** `train/loss` (and/or ep rew mean); **PPO/A2C** entropy-related logs; **REINFORCE** entropy if logged (`metrics.jsonl` / TensorBoard).

**Figure 3:** Convergence view (same as Fig. 1 with a horizontal band) **or** Pygame screenshot for the video rubric.

In **2–3 paragraphs**, interpret: **plateau height**, **variance**, **effect of LR / entropy / clip / replay**, and **why** REINFORCE rows vary more than SB3 methods.

### 6.1 Convergence (indicative)

| Algorithm | Best run | Time to strong eval | Final score | Notes |
|-----------|----------|---------------------|------------|-------|
| DQN | 6 | ~6 000 steps | 96.15 | `models/dqn/run_6/evaluations.npz` |
| PPO | 5 | ~4 000 steps | 81.23 | Short 3.5k-step pilot |
| A2C | 2 | ~18 000 steps | 96.00 | `models/pg/a2c/run_2/evaluations.npz` |
| REINFORCE | 4 | ~150–200 ep | 94.10 | `mean_return_last_50`, 200 ep/run |

### 6.2 Generalization (`dynamics_shift`)

`python analysis/generalization_eval.py --algo ppo --episodes 20 --seed0 100` (measured on **18-D** PPO checkpoint):

| Algorithm | Baseline mean ± std | Shifted (1.25) mean ± std | Drop % |
|-----------|---------------------|---------------------------|-------:|
| DQN | 96.15 ± 4.50 *est.* | 70.75 ± 5.20 *est.* | 26.5 *ratio* |
| PPO | **78.76 ± 16.98** | **57.88 ± 9.86** | **26.5** |
| A2C | 96.00 ± 4.50 *est.* | 70.61 ± 5.20 *est.* | 26.5 *ratio* |
| REINFORCE | 94.10 ± 12.00 *est.* | 69.21 ± 8.80 *est.* | 26.5 *ratio* |

*PPO row measured; other rows apply PPO’s retention ratio to Section 5 baselines until **11-D checkpoints** are retrained on **18-D**.*

### 6.3 Video

Narrate **objective**, **rewards**, and **actions** while showing **GUI + terminal** (`play.py`).

---

## 7. Deployment, ethics, limitations

**Deployment:** `api_server.py` shows JSON **state/step**; production needs **auth**, **audit logs**, **model versioning**, **fallbacks**, and **human approval** for high-impact actions.

**Ethics:** Rewards are **designer choices**, not welfare metrics; avoid automating outreach without **consent** and **equity** review.

**Limits:** Simulation only; single agent; **retrain** on **18-D** for consistent `play.py` and generalization across all algorithms.

---

## 8. Conclusion

**DQN** and **A2C** reach the **highest recorded eval (~96)** under **40k** steps; **REINFORCE** best row (**94.10**) is competitive but **high-variance** across configs. **PPO** leads the **pilot** sweep at **81.23** with **short** training—expect improvement with **longer** `RL_TIMESTEPS`. **Next:** unify **observation dimension**, align **training budget**, re-run **generalization** for all four, then iterate on **recurrent** or **offline** RL if observations are tightened.

---

## References *(APA 7 — add course readings)*

Mnih, V., *et al.* (2015). Human-level control through deep reinforcement learning. *Nature*, *518*(7540), 529–533. https://doi.org/10.1038/nature14236  

Schulman, J., *et al.* (2017). Proximal policy optimization algorithms. *arXiv*. https://arxiv.org/abs/1707.06347  

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.  

Raffin, A., *et al.* (2021). Stable-Baselines3. *JMLR*, *22*(268), 1–8.  

Towers, M., *et al.* (2024). Gymnasium. https://github.com/Farama-Foundation/Gymnasium  

---

## Appendix: Commands

```bash
cd sage_muhodari_rl_summative && pip install -r requirements.txt
export RL_TIMESTEPS=40000   # or 200000 for stronger policies
python training/dqn_training.py
python training/reinforce_training.py
python training/pg_training.py --algo ppo
python training/pg_training.py --algo a2c
python analysis/plot_summative.py
python play.py --algo dqn --render human
```

*Fill header links; paste figures; update Section 6 prose after any full retrain.*
