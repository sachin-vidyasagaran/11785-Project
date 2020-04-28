import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import pdb
from models import *
from utils import *

class DDPGagent:
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-3, critic_learning_rate=1e-3, gamma=0.99, tau=0.05, max_memory_size=50000):
        # Params
        self.num_states = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
        #self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.actor = self.actor.to(self.device)
        self.actor_target = self.actor_target.to(self.device)
        self.critic = self.critic.to(self.device)
        self.critic_target = self.critic_target.to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state):
        state = np.hstack((state['observation'], state['desired_goal']))

        state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        state = state.to(self.device)
        action = self.actor.forward(state)
        action = action.to('cpu')
        action = action.detach().numpy()[0]
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        for i in range(len(states)):
            states[i] = np.hstack((states[i]['observation'], states[i]['desired_goal']))
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        #print(rewards)
        for i in range(len(next_states)):
            next_states[i] = np.hstack((next_states[i]['observation'], next_states[i]['desired_goal']))
        next_states = torch.FloatTensor(next_states)

        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = states.to(self.device)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        Qvals = Qvals.to('cpu')
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        next_Q = next_Q.to('cpu')
        Qprime = rewards + self.gamma * next_Q
        lowbound = -1/(1 - self.gamma)
        Qprime = torch.clamp(Qprime, lowbound, 0)
        critic_loss = self.critic_criterion(Qvals, Qprime)
        #print("q val:", Qvals)
        #print("target q val td:", Qprime)
        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))