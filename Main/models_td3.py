import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import pdb
from torch.autograd import Variable
import os

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,td4=False):
        super(Critic, self).__init__()
        name = 'TD3' if td4==False else 'TD4'
        self.checkpoint_file = os.path.join('saved_models/',name+'_Critic')
        self.td4=td4
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear5 = nn.Linear(input_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.linear7 = nn.Linear(hidden_size, hidden_size)
        self.linear8 = nn.Linear(hidden_size, 1)

        if td4:
            self.linear9 = nn.Linear(input_size, hidden_size)
            self.linear10 = nn.Linear(hidden_size, hidden_size)
            self.linear11 = nn.Linear(hidden_size, hidden_size)
            self.linear12 = nn.Linear(hidden_size, 1)

    def forward(self, state, action,):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        q1 = self.linear4(x)

        x = torch.cat([state, action], 1)
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        x = F.relu(self.linear7(x))
        q2 = self.linear8(x)

        if self.td4:
            x = torch.cat([state, action], 1)
            x = F.relu(self.linear9(x))
            x = F.relu(self.linear10(x))
            x = F.relu(self.linear11(x))
            q3 = self.linear12(x)

            return q1, q2,q3
        else:
            return q1,q2

    def Q1(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        q1 = self.linear4(x)
        return q1

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Actor(nn.Module):
    def __init__(self, input_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.checkpoint_file = os.path.join('saved_models/','TD_Actor')
        self.linear1 = nn.Linear(input_size, 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
