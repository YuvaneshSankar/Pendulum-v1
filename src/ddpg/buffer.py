import numpy as np
import torch

class Buffer:
    def __init__(self, max_size, action_dim, state_dim):
        self.max_size = max_size
        self.ptr = 0  # pointer to the current position in buffer
        self.size = 0  # current size of the buffer
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.next_states = np.zeros((max_size, state_dim))
        self.dones = np.zeros((max_size, 1))


    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return dict(
            state=torch.FloatTensor(self.states[ind]),
            action=torch.FloatTensor(self.actions[ind]),
            reward=torch.FloatTensor(self.rewards[ind]),
            next_state=torch.FloatTensor(self.next_states[ind]),
            done=torch.FloatTensor(self.dones[ind])
        )
    
    def len(self):
        return self.size
    

    def ready(self, batch_size):
        """Check if buffer has enough samples"""
        return self.size >= batch_size