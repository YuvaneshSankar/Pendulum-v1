import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(rewards)), rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Episode Rewards")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_losses(actor_losses, critic_losses, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
