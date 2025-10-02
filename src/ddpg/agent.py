import torch.nn as nn
import torch
from network import Actor, Critic
from pathlib import Path
from buffer import Buffer
from noise import OUNoise

class Agent:
    def __init__(self, input_dim, output_dim, action_dim, max_action, actor_lr, critic_lr, gamma, tau, buffer_size=1000000, batch_size=64, noise_params=None):
        self.actor = Actor(input_dim, output_dim, max_action, lr=actor_lr)
        self.critic = Critic(input_dim, action_dim, lr=critic_lr)
        self.target_actor = Actor(input_dim, output_dim, max_action, lr=actor_lr)
        self.target_critic = Critic(input_dim, action_dim, lr=critic_lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        # Initialize replay buffer
        self.buffer = Buffer(buffer_size, action_dim, input_dim)
        # Initialize Ornstein-Uhlenbeck noise for exploration
        if noise_params is None:
            noise_params = {"mu": 0.0, "theta": 0.15, "sigma": 0.2}
        self.noise = OUNoise(action_dim, **noise_params)

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().detach().numpy()[0]
        if add_noise:
            action += self.noise.sample()
        return action

    def reset_noise(self):
        # Call this at the start of each episode
        self.noise.reset()

    def store_transition(self, state, action, reward, next_state, done):
        # Store experience in the replay buffer
        self.buffer.store(state, action, reward, next_state, done)

    def evaluate(self, state, action):
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        q_value = self.critic(state, action)
        return q_value

    def calculate_target(self, reward, next_state, done):
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        target_action = self.target_actor(next_state)
        yi = reward + (self.gamma * self.target_critic(next_state, target_action) * (1 - done))
        return yi

    def critic_loss(self, current_q, y_i):
        critic_loss = nn.MSELoss()(current_q, y_i)
        return critic_loss

    def actor_loss(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        quality = self.critic(state, action)
        #We have to maximize the quality but the optimizer in the nn minimizes the loss function so we do this -ve
        #Also we have to just give the qaulit*action to the optimizer it calculates the gradients of both
        #We take the mean becuase we have to go for the average quality of the batch
        actor_loss = -quality.mean()
        return actor_loss

    def update_online_networks(self, actor_loss, critic_loss):
        actor_loss_value = self.actor.weight_updates(actor_loss)
        critic_loss_value = self.critic.weight_updates(critic_loss)
        return actor_loss_value, critic_loss_value

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self):
        # Only learn if buffer has enough samples
        if not self.buffer.ready(self.batch_size):
            return None, None
        batch = self.buffer.sample(self.batch_size)
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)
        # Critic update
        current_q = self.critic(state, action)
        with torch.no_grad():
            target_action = self.target_actor(next_state)
            y_i = reward + (self.gamma * self.target_critic(next_state, target_action) * (1 - done))
        critic_loss = nn.MSELoss()(current_q, y_i)
        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        # Update networks
        self.update_online_networks(actor_loss, critic_loss)
        # Update target networks
        self.update_target_networks()
        return actor_loss.item(), critic_loss.item()

    def save_models(self, checkpoint_name="ddpg_checkpoint"):
        current_path = Path(__file__).resolve()
        project_root = current_path.parent.parent
        save_dir = project_root / "models" / "saved_models"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), save_dir / f"{checkpoint_name}_actor.pth")
        torch.save(self.critic.state_dict(), save_dir / f"{checkpoint_name}_critic.pth")
        torch.save(self.target_actor.state_dict(), save_dir / f"{checkpoint_name}_target_actor.pth")
        torch.save(self.target_critic.state_dict(), save_dir / f"{checkpoint_name}_target_critic.pth")

    def load_models(self, checkpoint_name="ddpg_checkpoint"):
        current_path = Path(__file__).resolve()
        project_root = current_path.parent.parent
        save_dir = project_root / "models" / "saved_models"
        self.actor.load_state_dict(torch.load(save_dir / f"{checkpoint_name}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(save_dir / f"{checkpoint_name}_critic.pth", map_location=self.device))
        self.target_actor.load_state_dict(torch.load(save_dir / f"{checkpoint_name}_target_actor.pth", map_location=self.device))
        self.target_critic.load_state_dict(torch.load(save_dir / f"{checkpoint_name}_target_critic.pth", map_location=self.device))
