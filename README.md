# Imitation Learning: BC vs IQL on Robotic Manipulation

**TL;DR:** Behavior Cloning (BC) and Implicit Q-Learning (IQL) are trained on identical expert demonstrations for the robosuite Lift task. IQL achieves higher overall success rate (80% vs 74%) and consistently higher reward under distribution shift, while BC is more precise within its training distribution. Full results and reproduction steps below.

---

## Quick Start

```bash
# 1. Setup
conda create -n robosuite_env python=3.9 && conda activate robosuite_env
pip install -r requirements.txt

# 2. Download dataset (place in ./data/)
#    https://robomimic.github.io/docs/datasets/robomimic_v0.1.html

# 3. Train
python -m robomimic.scripts.train --config configs/bc_lift.json
python -m robomimic.scripts.train --config configs/iql_lift.json

# 4. Evaluate (replace with your checkpoint paths)
python scripts/bc_rollout.py --ckpt outputs/bc_lift/<run>/models/model_epoch_2000.pth
python scripts/iql_rollout.py --ckpt outputs/iql_lift/<run>/models/model_epoch_2000.pth
python scripts/generalization_test.py \
    --bc_ckpt outputs/bc_lift/<run>/models/model_epoch_2000.pth \
    --iql_ckpt outputs/iql_lift/<run>/models/model_epoch_2000.pth
```

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

The dataset is not included in this repository due to its size. It can be downloaded from the [robomimic dataset page](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html) and should be placed in `./data/`.

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

## Reproducing the Experiments

### Environment setup

```bash
conda create -n robosuite_env python=3.9
conda activate robosuite_env
pip install -r requirements.txt
```

### Dataset

Download the Lift proficient-human dataset from the [robomimic dataset page](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html) and place it at `./data/low_dim_v141.hdf5`.

### Training

```bash
python -m robomimic.scripts.train --config configs/bc_lift.json
python -m robomimic.scripts.train --config configs/iql_lift.json
```

Both configs have `rollout.enabled` set to `False` to avoid a `mujoco_py` dependency conflict on newer MuJoCo versions. Evaluation is performed separately via the scripts below.

### Evaluation

```bash
# BC rollout (50 episodes)
python scripts/bc_rollout.py --ckpt <path_to_bc_checkpoint.pth>

# IQL rollout (50 episodes)
python scripts/iql_rollout.py --ckpt <path_to_iql_checkpoint.pth>

# Generalization test (3 seed groups × 20 episodes per model)
python scripts/generalization_test.py \
    --bc_ckpt <path_to_bc_checkpoint.pth> \
    --iql_ckpt <path_to_iql_checkpoint.pth>
```

Model checkpoints (`.pth` files) are not included due to file size. Training from the provided configs and dataset fully reproduces the reported results.

---

## Repository Structure

```
Imitation-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── bc_lift.json           # BC training config
│   └── iql_lift.json          # IQL training config
├── scripts/
│   ├── bc_rollout.py          # BC evaluation (--ckpt, --n_rollouts)
│   ├── iql_rollout.py         # IQL evaluation (--ckpt, --n_rollouts)
│   └── generalization_test.py # Cross-initialization test (--bc_ckpt, --iql_ckpt)
└── results/
    ├── bc_vs_iql_generalization.png
    └── bc_vs_iql_summary_table.png
```

---

## Environment

| Property | Value |
|---|---|
| Platform | macOS, Apple Silicon (arm64) |
| Python | 3.9.25 |
| robosuite | 1.4.1 |
| robomimic | 0.3.0 |
| MuJoCo | 3.2.0 |
