import torch.nn as nn
import torch.nn.functional as F
import torch
# input is given with parameters input dim, action dim, list of hidden sizes
class Actor(nn.Module):
    def __init__(self, num_inputs, action_dim, goals_dim, hidden_sizes):
        super(Actor, self).__init__()

        self.nlayers = len(hidden_sizes)
        self.input_size=num_inputs
        self.action_dim =action_dim
        seq = []

        for i in range(self.nlayers):
            if i == 0:
                seq.append(nn.Linear(num_inputs+goals_dim, hidden_sizes[i]))
            else:
                seq.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            seq.append(nn.ReLU())
#
        self.seq = nn.Sequential(*seq)
        print(*seq)
        self.outer = nn.Linear(hidden_sizes[self.nlayers-1], action_dim)
        self.tanh = nn.Tanh()

    def forward(self, inputs,goals):
        x = inputs
        x = torch.cat([x, goals], 1)
        x = self.seq(x)
        x = self.outer(x)
        x = self.tanh(x)

        return x

# input is given as state_dim,action_dim,hidden size list,q_dim(default:1) if applicable
class Critic(nn.Module):
    def __init__(self,state_dim, action_dim,goal_dim,hidden_sizes):
        super(Critic, self).__init__()
        self.action_space = action_dim
        self.state_space = state_dim
        self.hidden_sizes=hidden_sizes
        self.nlayers = len(hidden_sizes)

        seq = []

        for i in range(self.nlayers):
            if i==0:
                linear = nn.Linear(state_dim+action_dim+goal_dim, hidden_sizes[i])
            else:
                linear = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            seq.append(linear)
            seq.append(nn.ReLU())

        self.seq=nn.Sequential(*seq)
        self.outer=nn.Linear(hidden_sizes[self.nlayers-1], 1)



    def forward(self, states, actions,goals):
        s = states
        a = actions
        x=torch.cat([s,a,goals],1)
        x = self.seq(x)
        # x = F.relu(x)
        x = self.outer(x)
#         x = self.relu(x)

        return x
