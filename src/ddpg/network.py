import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, max_action, hidden_dim=400, lr=1e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.max_action = max_action

        # These lines initialize the final layer weights and biases to small random values between -0.003 and 0.003. This is a crucial trick in DDPG for stable learning.
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Use tanh to ensure output is in [-1, 1]
        return self.max_action * x  # as the tanh output is between -1 and 1, we scale it to the action range which is [-max_action, max_action]
    
    def weight_updates(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400, lr=1e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1) #concatenate state and action tensors
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def weight_updates(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
