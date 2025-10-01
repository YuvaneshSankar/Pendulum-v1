import torch.nn as nn
import torch 
from network import Actor, Critic
from pathlib import Path


class Agent:
    def __init__(self,input_dim,output_dim,action_dim,max_action,actor_lr,critic_lr,gamma,tau):
        self.actor=Actor(input_dim,output_dim,max_action,lr=actor_lr)
        self.critic=Critic(input_dim,action_dim,lr=critic_lr)
        self.target_actor=Actor(input_dim,output_dim,max_action,lr=actor_lr)
        self.target_critic=Critic(input_dim,action_dim,lr=critic_lr)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)
        self.gamma=gamma
        self.tau=tau

    def select_action(self,state):
        state=torch.FloatTensor(state).unsqueeze(0)
        action=self.actor(state)
        return action
    
    def evaluate(self,state,action):
        state=torch.FloatTensor(state).to(self.device)
        action=torch.FloatTensor(action).to(self.device)
        q_value=self.critic(state,action)
        return q_value


    def calculate_target(self,reward,next_state,done):
        reward=torch.FloatTensor(reward).to(self.device)
        next_state=torch.FloatTensor(next_state).to(self.device)
        target_action=self.target_actor(next_state)
        yi=reward + (self.gamma * self.target_critic(next_state,target_action) * (1-done))
        return yi

    def critic_loss(self,current_q,y_i):
        critic_loss=nn.MSELoss()(current_q,y_i)
        return critic_loss
    
    def actor_loss(self,state):
        state=torch.FloatTensor(state).to(self.device)
        action=self.actor(state)
        quality=self.critic(state,action)
        #We have to maximize the quality but the optimizer in the nn minimizes the loss function so we do this -ve 
        #Also we have to just give the qaulit*action to the optimizer it calculates the gradients of both
        #We take the mean becuase we have to go for the average quality of the batch
        actor_loss=-quality.mean()
        return actor_loss
    
    def update_online_networks(self,actor_loss,critic_loss):
        actor_loss_value=self.actor.weight_updates(actor_loss)
        critic_loss_value=self.critic.weight_updates(critic_loss)
        return actor_loss_value,critic_loss_value
    
    def update_target_networks(self):
        for target_param,param in zip(self.target_actor.parameters(),self.actor.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        
        for target_param,param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def save_models(self,checkpoint_name="ddpg_checkpoint"):

        current_path=Path(__file__).resolve()
        project_root=current_path.parent.parent
        save_dir=project_root/"models"/"saved_models"
        save_dir.mkdir(parents=True,exist_ok=True)
        torch.save(self.actor.state_dict(),save_dir/f"{checkpoint_name}_actor.pth")
        torch.save(self.critic.state_dict(),save_dir/f"{checkpoint_name}_critic.pth")
        torch.save(self.target_actor.state_dict(),save_dir/f"{checkpoint_name}_target_actor.pth")
        torch.save(self.target_critic.state_dict(),save_dir/f"{checkpoint_name}_target_critic.pth")


    