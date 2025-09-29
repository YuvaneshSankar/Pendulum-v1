import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,actor_lr):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,output_dim)
        self.optimizer=optim.Adam(self.parameters(),lr=actor_lr)


    def forward(self,state):
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x), dim=-1)
        return x  
    
    def weight_updates(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
class Critic(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,critic_lr):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,output_dim)
        self.optimizer=optim.Adam(self.parameters(),lr=critic_lr)
        
    def forward(self,state,action):
        x=torch.cat([state,action],dim=1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
    def weight_updates(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()