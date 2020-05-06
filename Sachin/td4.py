import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import pdb
from models_td3 import *
from utils import *


class TD4(object):
    def __init__(self, env, hidden_size=256, actor_learning_rate=3e-4, critic_learning_rate=3e-4, gamma=0.99, tau=0.005,
                 max_memory_size=50000, policy_freq=2):
        self.num_states = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
        self.desired = env.observation_space['desired_goal'].shape[0]
        # self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Networks
        self.actor = Actor(self.num_states, self.num_actions)
        self.actor_target = Actor(self.num_states, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions,td4=True)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions,td4=True)
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
        self.statenorm = Normalizer(self.num_states, 0)
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.total_it = 0
        self.policy_freq = policy_freq

    def get_action(self, state):
        state = np.hstack((state['observation'], state['desired_goal']))
        state = self.statenorm.normalize(state, batch=False)

        state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        state = state.to(self.device)
        self.actor.eval()
        action = self.actor.forward(state)
        action = action.to('cpu')
        action = action.detach().numpy()[0]
        return action

    def update(self, batch_size):
        self.total_it += 1
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        for i in range(len(states)):
            states[i] = np.hstack((states[i]['observation'], states[i]['desired_goal']))
        states = self.statenorm.normalize(states)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)

        # print(rewards)
        for i in range(len(next_states)):
            next_states[i] = np.hstack((next_states[i]['observation'], next_states[i]['desired_goal']))
        next_states = self.statenorm.normalize(next_states)
        next_states = torch.FloatTensor(next_states)

        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)

        # Critic loss
        Qval1, Qval2,Qval3 = self.critic.forward(states, actions)
        Qval1 = Qval1.to('cpu')
        Qval2 = Qval2.to('cpu')
        Qval3 = Qval3.to('cpu')

        action_noise = torch.as_tensor(np.random.normal(loc=0, scale=0.1, size=actions.size()), device=self.device)
        next_actions = self.actor_target.forward(next_states)

        next_actions += action_noise
        # TODO: clip from action low and action high
        next_actions = torch.clamp(next_actions, -1, 1)

        target_Q1, target_Q2,target_Q3 = self.critic_target.forward(next_states, next_actions.detach())
        target_Q = torch.min(target_Q1, torch.min(target_Q2,target_Q3))
        target_Q = target_Q.to('cpu')

        Qprime = rewards + self.gamma * target_Q

        # lowbound = -1 / (1 - self.gamma)
        # Qprime = torch.clamp(Qprime, lowbound, 0)
        critic_loss = self.critic_criterion(Qval1, Qprime) + self.critic_criterion(Qval2, Qprime)+self.critic_criterion(Qval3, Qprime)
        # print('critic_loss:', critic_loss.item())
        # Actor loss
        # curr_actions = self.actor.forward(states)

        # policy_loss += 1 * (curr_actions).pow(2).mean()
        # print('policy loss:', policy_loss.item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:

            policy_loss = -self.critic.Q1(states, self.actor.forward(states)).mean()

            # update networks
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        self.statenorm.save_normalizer()

    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()