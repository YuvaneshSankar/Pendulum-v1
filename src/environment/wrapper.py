import gymnasium as gym
import numpy as np

class NormalizeActionWrapper(gym.ActionWrapper):
    """
    Gym wrapper to ensure actions are always in the environment's action space and optionally scale them.
    """
    def __init__(self, env):
        super(NormalizeActionWrapper, self).__init__(env)
        self.low = self.action_space.low
        self.high = self.action_space.high

    def action(self, action):
        # Clip and scale actions to valid range
        return np.clip(action, self.low, self.high)

    def reverse_action(self, action):
        return np.clip(action, self.low, self.high)

# You can now use this wrapper in scripts like:
# env = NormalizeActionWrapper(gym.make("Pendulum-v1"))
