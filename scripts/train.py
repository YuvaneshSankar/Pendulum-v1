import os
import yaml
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

from ddpg.agent import Agent
from ddpg.noise import OUNoise
from environment.wrapper import NormalizeActionWrapper
from utils.logger import setup_logger
from utils.plotting import plot_rewards, plot_losses
from utils.common import set_seed

# Load config
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

env_name = config["environment"]["name"]
max_episode_steps = config["environment"]["max_episode_steps"]
seed = 42
set_seed(seed)

# Logger
logger = setup_logger()

# Environment
env = NormalizeActionWrapper(gym.make(env_name))
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Agent
agent = Agent(
    input_dim=state_dim,
    output_dim=action_dim,
    action_dim=action_dim,
    max_action=max_action,
    actor_lr=config["agent"]["lr_actor"],
    critic_lr=config["agent"]["lr_critic"],
    gamma=config["agent"]["gamma"],
    tau=config["agent"]["tau"],
    buffer_size=config["agent"]["buffer_size"],
    batch_size=config["agent"]["batch_size"],
    noise_params=config["noise"]
)

total_timesteps = config["training"]["total_timesteps"]
warmup_steps = config["training"]["warmup_steps"]
update_frequency = config["training"]["update_frequency"]
save_frequency = config["training"]["save_frequency"]
log_frequency = config["logging"]["log_frequency"]

rewards_history = []
actor_losses = []
critic_losses = []

obs, _ = env.reset()
episode_reward = 0
episode = 0
timestep = 0

while timestep < total_timesteps:
    agent.reset_noise()
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    for step in range(max_episode_steps):
        if timestep < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, add_noise=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, float(done))
        episode_reward += reward
        obs = next_obs
        timestep += 1

        if timestep >= warmup_steps and timestep % update_frequency == 0:
            actor_loss, critic_loss = agent.learn()
            if actor_loss is not None:
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        if timestep % save_frequency == 0:
            agent.save_models(f"ddpg_checkpoint_{timestep}")
            logger.info(f"Checkpoint saved at timestep {timestep}")

        if done or timestep >= total_timesteps:
            break

    rewards_history.append(episode_reward)
    episode += 1

    if episode % log_frequency == 0:
        logger.info(f"Episode {episode} | Timestep {timestep} | Reward: {episode_reward:.2f}")

# Save final model
agent.save_models("ddpg_final")
logger.info("Training complete. Final model saved.")

# Save rewards and losses
results_dir = Path(__file__).parent.parent / "results" / "plots"
results_dir.mkdir(parents=True, exist_ok=True)
np.savetxt(results_dir / "episode_rewards.txt", rewards_history)
np.savetxt(results_dir / "actor_losses.txt", actor_losses)
np.savetxt(results_dir / "critic_losses.txt", critic_losses)

# Plot
plot_rewards(rewards_history, save_path=results_dir / "training_rewards.png")
plot_losses(actor_losses, critic_losses, save_path=results_dir / "loss_curves.png")
