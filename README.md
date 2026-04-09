[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmahi97%2FIE579-GameTheory&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# IE579 — Game Theory and Multi-Agent Reinforcement Learning

Homework repository for **IE579: Game Theory and Multi-Agent Reinforcement Learning** at [KAIST](https://www.kaist.ac.kr).

> **Student:** Mohammad Mahdi Rahimi · `mahi@kaist.ac.kr`

---

## Table of Contents

- [Course Overview](#course-overview)
- [Repository Structure](#repository-structure)
- [Homework 1 — Nash Equilibrium & Correlated Equilibrium](#homework-1--nash-equilibrium--correlated-equilibrium)
- [Homework 2 — Repeated & Sequential Games + Graph Neural Networks](#homework-2--repeated--sequential-games--graph-neural-networks)
- [Homework 3 — Single-Agent RL for Traffic Control (SUMO)](#homework-3--single-agent-rl-for-traffic-control-sumo)
- [Homework 4 — Multi-Agent RL on Particle Environments](#homework-4--multi-agent-rl-on-particle-environments)
- [Final Project — XQMIX for StarCraft Multi-Agent Challenge](#final-project--xqmix-for-starcraft-multi-agent-challenge)
- [Requirements](#requirements)

---

## Course Overview

IE579 covers the theoretical foundations of game theory and their application to multi-agent reinforcement learning systems. Topics include:

- Normal-form and extensive-form games
- Nash equilibria (pure, mixed, and correlated)
- Repeated and sequential games
- Single-agent and multi-agent reinforcement learning
- Graph-based neural architectures for cooperative agents

---

## Repository Structure

```
IE579-GameTheory/
├── HW1/          # Nash Equilibrium & Correlated Equilibrium
├── HW2/          # Repeated/Sequential Games + GNN
├── HW3/          # Single-Agent RL for Traffic Control (SUMO)
├── HW4/          # Multi-Agent RL — Particle Environment (DGN)
└── README.md
```

---

## Homework 1 — Nash Equilibrium & Correlated Equilibrium

**Topics:** Pure Nash, Mixed Nash, Correlated Nash Equilibrium, Best-Response Analysis, Security Strategies

### Contents

| File | Description |
|------|-------------|
| `main.tex` | LaTeX source for all solutions |
| `Solutions.pdf` | Compiled PDF of written solutions |

### Key Problems

1. **Bidding Game** — Formulated a two-player game, proved (50, 50) is the unique Nash Equilibrium via best-response analysis.
2. **Congestion Game** — Analysed a 4,000-player road network; showed how adding a zero-cost road shifts the Nash outcome from 65 to 80 minutes (Braess's Paradox).
3. **Security Strategies** — Proved equivalence between saddle-point entries, security strategies, and Nash equilibria.
4. **Cournot Competition** — Derived symmetric Nash for an $n$-firm oligopoly: $q^* = \frac{a-c}{n+1}$.
5. **Pure & Mixed Nash** — Found both pure Nash outcomes and the unique mixed-strategy equilibrium; computed expected payoffs.
6. **Correlated Equilibrium** — Derived the full family of correlated equilibria and identified the welfare-maximising distribution ($P_c = \frac{1}{3}$).
7. **Best-Response Algorithm** — Disproved convergence in general $n\times m$ games and proposed a divide-and-conquer algorithm with time complexity $O(MN^2)$ worst-case and $O(MN)$ best-case.

---

## Homework 2 — Repeated & Sequential Games + Graph Neural Networks

**Topics:** Subgame-perfect equilibrium, backward induction, folk theorem, Graph Neural Networks (GNN)

### Contents

| File | Description |
|------|-------------|
| `main.tex` | LaTeX source for theoretical solutions |
| `GNN.ipynb` | Jupyter notebook: GNN implementation and tuning |

### Key Problems

- Analysis of Nash equilibria in repeated and sequential games using backward induction and the folk theorem.
- Implementation and hyperparameter tuning of a **Graph Neural Network** from scratch in PyTorch.

---

## Homework 3 — Single-Agent RL for Traffic Control (SUMO)

**Topics:** Deep Q-Network (DQN), Deep Deterministic Policy Gradient (DDPG), Traffic simulation

### Contents

| File | Description |
|------|-------------|
| `HW3_DQN.ipynb` | DQN agent for traffic light control |
| `HW3_DDPG.ipynb` | DDPG agent for traffic light control |
| `SUMO/TrafficEnv_DQN.py` | Custom Gym environment (DQN) wrapping SUMO |
| `SUMO/TrafficEnv_DDPG.py` | Custom Gym environment (DDPG) wrapping SUMO |
| `SUMO/Single_Intersection.*` | SUMO network, route, and configuration files |
| `Visualization/` | Training curves and result plots |

### Environment

A single-intersection traffic network simulated in [SUMO](https://eclipse.dev/sumo/). A single RL agent controls the traffic light phase to minimise vehicle waiting time.

| Algorithm | Observation | Action Space |
|-----------|-------------|--------------|
| DQN | Queue lengths per lane | Discrete (phase selection) |
| DDPG | Queue lengths per lane | Continuous (phase duration) |

### Requirements

```bash
pip install traci sumo gym torch
```

---

## Homework 4 — Multi-Agent RL on Particle Environments

**Topics:** Graph Convolutional Reinforcement Learning, cooperative multi-agent settings

### Contents

| File | Description |
|------|-------------|
| `DGN.ipynb` | Full DGN implementation and training |
| `simple_spread.py` | Modified `simple_spread` scenario from `multiagent-particle-envs` |
| `models/` | Saved model checkpoints (`DGN-R0`, `DGN-R1`, `DGN-R5`) |
| `vis/` | Training visualisations |

### Algorithm

Implements **DGN — Graph Convolutional Reinforcement Learning** ([Jiang et al., ICLR 2020](https://arxiv.org/pdf/1810.09202.pdf)), which encodes agent neighbourhoods with graph convolution and learns cooperative policies end-to-end.

### Environment

OpenAI **Simple Spread** (`multiagent-particle-envs`): $N$ cooperative agents must cover $N$ landmarks while avoiding collisions.

### Requirements

```bash
pip install git+https://github.com/openai/multiagent-particle-envs
pip install "gym==0.10.5"   # downgrade required for compatibility
pip install torch
```

---

## Final Project — XQMIX for StarCraft Multi-Agent Challenge

An extended version of **QMIX** ([Rashid et al., ICML 2018](https://arxiv.org/abs/1803.11485)) applied to the [StarCraft Multi-Agent Challenge (SMAC)](https://github.com/oxwhirl/smac).

> **Repository:** [github.com/mahi97/XQMIX](https://github.com/mahi97/XQMIX)

XQMIX augments the standard QMIX monotonic value-factorisation with additional structural constraints and auxiliary objectives to improve sample efficiency and final win rates on hard SMAC scenarios.

---

## Requirements

The table below summarises the primary dependencies across all homeworks.

| Homework | Key Dependencies |
|----------|-----------------|
| HW1 | LaTeX (`pdflatex`, `minted`, `pgfplots`, `tikz`) |
| HW2 | PyTorch, Jupyter |
| HW3 | SUMO ≥ 1.8, `traci`, OpenAI Gym, PyTorch |
| HW4 | `multiagent-particle-envs`, Gym 0.10.5, PyTorch |
| Final | SMAC, PyTorch, see [XQMIX](https://github.com/mahi97/XQMIX) |

---

*KAIST · IE579 · Spring 2021*
