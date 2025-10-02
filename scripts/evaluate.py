import gymnasium as gym
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from ddpg.agent import Agent
from environment.wrapper import NormalizeActionWrapper

# Settings
env_name = "Pendulum-v1"
episodes = 20

# Environment
env = NormalizeActionWrapper(gym.make(env_name, render_mode="human"))
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Agent
agent = Agent(
    input_dim=state_dim,
    output_dim=action_dim,
    action_dim=action_dim,
    max_action=max_action,
    actor_lr=1e-3,
    critic_lr=1e-3,
    gamma=0.99,
    tau=0.005
)
agent.load_models("ddpg_final")

# Evaluation loop
rewards = []
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(obs, add_noise=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    rewards.append(total_reward)
    print(f"Eval Episode {ep+1}: Reward = {total_reward:.2f}")

print(f"Average Reward over {episodes} episodes: {np.mean(rewards):.2f}")
env.close()
