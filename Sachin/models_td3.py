import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import pdb
from torch.autograd import Variable


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear5 = nn.Linear(input_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.linear7 = nn.Linear(hidden_size, hidden_size)
        self.linear8 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
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

        return q1, q2

    def Q1(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        q1 = self.linear4(x)
        return q1


class Actor(nn.Module):
    def __init__(self, input_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
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