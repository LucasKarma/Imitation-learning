# P2 — Imitation Learning: BC vs IQL on Robotic Manipulation

A comparative study of Behavior Cloning (BC) and Implicit Q-Learning (IQL) on the **Lift** task using [robosuite](https://robosuite.ai/) and [robomimic](https://robomimic.github.io/). Both algorithms are trained on identical expert demonstration data and evaluated across overall success rate, average reward, and cross-initialization generalization.

---

## Task

The **Lift** task requires a Panda robot arm to locate a cube on a tabletop and lift it above a target height. The robot is controlled via an OSC_POSE controller (7-dimensional end-effector delta commands). Observations are 42-dimensional, combining proprioceptive state and object state.

---

## Algorithms

**BC (Behavior Cloning)** is a supervised learning approach that directly regresses the expert's action at each timestep. It is simple and fast to train, but its performance is bounded by the quality and coverage of the demonstration data. When the agent encounters states not seen during training, prediction errors compound — a phenomenon known as covariate shift.

**IQL (Implicit Q-Learning)** is an offline reinforcement learning algorithm that learns a Q-function and a separate value function (V-function) from demonstration data, without ever querying actions outside the dataset. By replacing the standard max operator in the Bellman update with an expectile regression over the V-function, IQL avoids overestimating the value of out-of-distribution actions — the central challenge of offline RL. This allows it to reason about long-term return rather than simply imitating observed actions.

---

## Dataset

Both models are trained on the official robomimic **proficient-human (ph)** dataset for the Lift task.

| Property | Value |
|---|---|
| File | `low_dim_v141.hdf5` |
| Trajectories | 200 expert demonstrations |
| Steps per trajectory (avg) | ~59 |
| Observation dimension | 42 |
| Action dimension | 7 (OSC_POSE) |

The dataset is not included in this repository due to its size. It can be downloaded from the [robomimic dataset page](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html).

---

## Results

### Experiment 1 — Overall evaluation (50 rollouts)

| Metric | BC | IQL |
|---|---|---|
| Success rate | 74% (37/50) | **80% (40/50)** |
| Avg reward | not recorded | 13.26 |
| Training epochs | 2000 | 2000 |

### Experiment 2 — Generalization test (3 seed groups × 20 rollouts each)

Three independent seed groups were used to evaluate both models under different random initializations. BC and IQL were evaluated on identical initial conditions within each group.

| Scenario | BC SR | IQL SR | BC Reward | IQL Reward |
|---|---|---|---|---|
| Seed Group A (0–19) | **95%** | 90% | **14.44** | 11.76 |
| Seed Group B (100–119) | 85% | 85% | 11.43 | **17.00** |
| Seed Group C (200–219) | 80% | 80% | 9.92 | **12.92** |

**Key finding:** BC achieves higher success rate under seed group A (initializations closer to its training distribution), consistent with the covariate shift hypothesis. As initialization diversity increases, success rates converge, but IQL maintains consistently higher average reward — indicating better action quality in unfamiliar states even when task completion rates are equivalent.

### Result charts

![Generalization comparison](results/bc_vs_iql_generalization.png)

![Summary table](results/bc_vs_iql_summary_table.png)

---

## Reproducing the experiments

**Environment setup**

```bash
conda create -n robosuite_env python=3.9
conda activate robosuite_env
pip install robosuite robomimic "mujoco==3.2.0"
```

**Training**

```bash
# BC
python -m robomimic.scripts.train --config configs/bc_lift.json

# IQL
python -m robomimic.scripts.train --config configs/iql_lift.json
```

Both configs have `rollout.enabled` set to `False` to avoid a `mujoco_py` dependency conflict. Evaluation is performed separately via the scripts below.

**Evaluation**

```bash
# BC rollout (50 episodes)
python scripts/bc_rollout.py

# IQL rollout (50 episodes)
python scripts/iql_rollout.py

# Generalization test (3 seed groups × 20 episodes per model)
python scripts/generalization_test.py
```

Model checkpoints (`.pth` files) are not included due to file size. Training from the provided configs and dataset fully reproduces the reported results.

---

## Repository structure

```
p2-imitation-learning/
├── README.md
├── .gitignore
├── configs/
│   ├── bc_lift.json           # BC training config
│   └── iql_lift.json          # IQL training config
├── scripts/
│   ├── bc_rollout.py          # BC evaluation script (50 rollouts)
│   ├── iql_rollout.py         # IQL evaluation script (50 rollouts)
│   └── generalization_test.py # Cross-initialization generalization test
├── results/
│   ├── bc_vs_iql_generalization.png
│   └── bc_vs_iql_summary_table.png
└── logs/
    ├── 工程复盘日志_P2_BC训练_4_10下午.md
    └── 工程复盘日志_P2_BC_vs_IQL对比实验_4_10晚上.md
```

---

## Environment

| Property | Value |
|---|---|
| Platform | macOS, Apple Silicon (arm64) |
| Python | 3.9.25 |
| robosuite | latest |
| robomimic | 0.3.0 |
| MuJoCo | 3.2.0 |
