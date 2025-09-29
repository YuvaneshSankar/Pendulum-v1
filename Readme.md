# DDPG for Pendulum Control

A PyTorch implementation of Deep Deterministic Policy Gradient (DDPG) for continuous control tasks, demonstrated on the Pendulum-v1 environment from OpenAI Gym.

![Pendulum Environment](https://gymnasium.farama.org/_images/pendulum.gif)


## Table of Contents
- [DDPG for Pendulum Control](#ddpg-for-pendulum-control)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Features](#key-features)
  - [Algorithm](#algorithm)
    - [Key Components](#key-components)
    - [Update Equations](#update-equations)
  - [Environment](#environment)
    - [Pendulum-v1](#pendulum-v1)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
    - [Training](#training)
    - [Testing](#testing)
    - [Monitoring](#monitoring)
  - [Project Structure](#project-structure)
  - [Implementation Details](#implementation-details)
    - [Network Architecture](#network-architecture)
    - [Training Process](#training-process)
    - [Key Implementation Features](#key-implementation-features)
  - [Results](#results)
    - [Expected Performance](#expected-performance)
  - [Hyperparameters](#hyperparameters)
    - [Default Configuration](#default-configuration)
    - [Tuning Guidelines](#tuning-guidelines)
  - [Contributing](#contributing)
    - [Development Setup](#development-setup)
    - [Running Tests](#running-tests)
  - [License](#license)
  - [References](#references)
    - [Citation](#citation)

## Overview

This project implements the **Deep Deterministic Policy Gradient (DDPG)** algorithm, a model-free off-policy actor-critic algorithm that can learn policies in high-dimensional, continuous action spaces. The implementation is tested on the **Pendulum-v1** environment, where the goal is to swing up and balance an inverted pendulum.

### Key Features
- Clean, modular PyTorch implementation
- Configurable hyperparameters via YAML files
- Comprehensive logging with TensorBoard support
- Experience replay buffer with efficient sampling
- Ornstein-Uhlenbeck noise for exploration
- Soft target network updates
- Model checkpointing and evaluation scripts

## Algorithm

DDPG combines the benefits of:
- **DQN**: Experience replay and target networks for stability
- **Actor-Critic**: Separate networks for policy (actor) and value function (critic)
- **Policy Gradients**: Direct policy optimization for continuous actions

### Key Components
1. **Actor Network (μ)**: Maps states to actions
2. **Critic Network (Q)**: Estimates Q-values for state-action pairs
3. **Target Networks**: Slowly updated copies for stability
4. **Experience Replay**: Stores and samples past experiences
5. **Exploration Noise**: Ornstein-Uhlenbeck process for action exploration

### Update Equations

**Critic Update:**
L = 1/N ∑(Q(si, ai) - yi)²
where yi = ri + γ Q'(si+1, μ'(si+1))



**Actor Update:**
∇θμ J ≈ 1/N ∑ ∇a Q(s, a)|s=si,a=μ(si) ∇θμ μ(s)|s=si



**Target Network Update:**
θ' ← τθ + (1-τ)θ'



## Environment

### Pendulum-v1
- **State Space**: 3D continuous [cos(θ), sin(θ), angular_velocity]
- **Action Space**: 1D continuous torque [-2.0, 2.0]
- **Reward**: Negative reward based on angle and angular velocity
- **Episode Length**: 200 steps
- **Goal**: Keep pendulum upright with minimal control effort

The reward function encourages the pendulum to stay upright (θ ≈ 0) with low angular velocity and minimal torque application.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU training)

### Setup
1. **Clone the repository:**
git clone https://github.com/yourusername/ddpg-pendulum.git
cd ddpg-pendulum



2. **Create virtual environment:**
python -m venv ddpg_env
source ddpg_env/bin/activate # On Windows: ddpg_env\Scripts\activate



3. **Install dependencies:**
pip install -r requirements.txt



4. **Verify installation:**
python -c "import torch; import gym; print('Installation successful!')"



## Usage

### Training
Train a DDPG agent with default hyperparameters:
python scripts/train.py



Train with custom config:
python scripts/train.py --config config/custom_config.yaml



Train with specific parameters:
python scripts/train.py --lr_actor 0.0005 --lr_critic 0.001 --batch_size 128



### Testing
Test a trained model:
python scripts/test.py --model models/saved_models/ddpg_best.pth



Evaluate and render episodes:
python scripts/evaluate.py --model models/saved_models/ddpg_best.pth --episodes 10 --render



### Monitoring
View training progress with TensorBoard:
tensorboard --logdir logs/tensorboard



## Project Structure

```
ddpg_pendulum/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── config/
│   ├── config.yaml              # Default hyperparameters
│   └── custom_config.yaml       # Custom configurations
├── src/
│   ├── __init__.py
│   ├── ddpg/
│   │   ├── __init__.py
│   │   ├── agent.py             # DDPG agent implementation
│   │   ├── networks.py          # Actor and Critic networks
│   │   ├── buffer.py            # Experience replay buffer
│   │   └── noise.py             # Ornstein-Uhlenbeck noise
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py            # Training logger
│   │   ├── plotting.py          # Visualization utilities
│   │   └── common.py            # Common utilities
│   └── environment/
│       ├── __init__.py
│       └── wrapper.py           # Environment wrapper
├── scripts/
│   ├── train.py                 # Training script
│   ├── test.py                  # Testing script
│   └── evaluate.py              # Evaluation script
├── notebooks/
│   ├── explore_env.ipynb        # Environment exploration
│   └── analysis.ipynb           # Results analysis
├── experiments/
│   └── configs/                 # Experiment configurations
├── models/
│   └── saved_models/            # Trained model checkpoints
├── logs/
│   ├── tensorboard/             # TensorBoard logs
│   └── training/                # Training logs
└── results/
    ├── plots/                   # Training plots
    └── videos/                  # Episode recordings
```



## Implementation Details

### Network Architecture
- **Actor**: State → [400] → [300] → Action (tanh activation)
- **Critic**: (State, Action) → [400] → [300] → Q-value
- Both networks use ReLU activation in hidden layers

### Training Process
1. **Warmup Phase**: Random actions for initial exploration
2. **Experience Collection**: Actor selects actions with noise
3. **Batch Learning**: Sample from replay buffer and update networks
4. **Target Updates**: Soft updates to target networks
5. **Evaluation**: Periodic evaluation without noise

### Key Implementation Features
- **Gradient Clipping**: Prevents exploding gradients
- **Batch Normalization**: Stabilizes learning (optional)
- **Learning Rate Scheduling**: Adaptive learning rates
- **Early Stopping**: Based on evaluation performance

## Results

### Expected Performance
- **Training Time**: ~30-60 minutes on CPU, ~10-15 minutes on GPU
- **Sample Efficiency**: Converges within 50K-100K steps
- **Final Performance**: Average reward > -200 (pendulum upright)



## Hyperparameters

### Default Configuration
environment:
name: "Pendulum-v1"
max_episode_steps: 200

agent:
lr_actor: 0.001 # Actor learning rate
lr_critic: 0.002 # Critic learning rate
tau: 0.005 # Soft update rate
gamma: 0.99 # Discount factor
batch_size: 64 # Minibatch size
buffer_size: 1000000 # Replay buffer size

network:
hidden_size: # Hidden layer dimensions

noise:
type: "ou_noise" # Noise type
mu: 0.0 # Mean
theta: 0.15 # Mean reversion rate
sigma: 0.2 # Volatility

training:
total_timesteps: 100000 # Total training steps
warmup_steps: 1000 # Random exploration steps
update_frequency: 1 # Network update frequency
save_frequency: 10000 # Model save frequency



### Tuning Guidelines
- **Learning Rates**: Start with 1e-3 for actor, 1e-3 or 2e-3 for critic
- **Tau**: Small values (0.001-0.01) for stability
- **Noise**: Adjust sigma based on action space scale
- **Batch Size**: 64-256, larger for more stable gradients

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
git clone https://github.com/YuvaneshSankar/Pendulum-v1
cd ddpg-pendulum
pip install -e .



### Running Tests
python -m pytest tests/



## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## References

1. **DDPG Paper**: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
2. **OpenAI Spinning Up**: [DDPG Documentation](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
3. **Gym Environment**: [Pendulum-v1](https://gym.openai.com/envs/Pendulum-v1/)

### Citation
@article{lillicrap2015continuous,
title={Continuous control with deep reinforcement learning},
author={Lillicrap, Timothy P and Hunt, Jonathan J and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
journal={arXiv preprint arXiv:1509.02971},
year={2015}
}



---

**Questions or Issues?** Please open an issue or reach out to [your-email@domain.com]

**Star ⭐ this repo if you found it helpful!**