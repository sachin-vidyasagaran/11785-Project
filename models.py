import torch.nn as nn
import torch.nn.functional as F
# input is given with parameters input dim, action dim, list of hidden sizes
class Actor(nn.Module):
    def __init__(self, num_inputs, action_dim,hidden_sizes):
        super(Actor, self).__init__()

        self.nlayers = len(hidden_sizes)
        self.input_size=num_inputs
        self.action_dim =action_dim
        seq=[]

        for i in range(self.nlayers):
            if i==0:
                # linear = nn.Linear(num_inputs, hidden_sizes[i])
                seq.append(nn.Linear(num_inputs, hidden_sizes[i]))
            # elif i==self.nlayers-1:
            #     linear = nn.Linear(hidden_size[i - 1], hidden_sizes[i])
            else:
                # linear = nn.Linear(hidden_size[i-1], hidden_sizes[i])
                seq.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
#             print(*seq)
            # bn = nn.BatchNorm1d(hidden_sizes[i])
            # print(i)
            # seq.append(bn)
            seq.append(nn.ReLU())
#         print(*seq)
        self.seq=nn.Sequential(*seq)
        print(*seq)
        self.outer=nn.Linear(hidden_sizes[self.nlayers-1], action_dim)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        x=inputs
        x=self.seq(x)
        # x=F.relu(x)
        x=self.outer(x)
        x=self.tanh(x)

        return x

# input is given as state_dim,action_dim,hidden size list,q_dim(default:1) if applicable
class Critic(nn.Module):
    def __init__(self,state_dim, action_dim,hidden_sizes,q_dim=1):
        super(Critic, self).__init__()
        self.action_space = action_dim
        self.state_space = state_dim
        self.hidden_sizes=hidden_sizes
        self.nlayers = len(hidden_sizes)
        self.qdim=q_dim
        seq=[]

        for i in range(self.nlayers):
            if i==0:
                linear = nn.Linear(state_dim+action_dim, hidden_sizes[i])
            # elif i==self.nlayers-1:
            #     linear = nn.Linear(hidden_size[i - 1], hidden_sizes[i])
            else:
                linear = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
#             bn = nn.BatchNorm1d(hidden_sizes[i])
            seq.append(linear)
#             seq.append(bn)
            seq.append(nn.ReLU())
        self.seq=nn.Sequential(*seq)
        self.outer=nn.Linear(hidden_sizes[self.nlayers-1], q_dim)
#         self.relu = nn.ReLU()



    def forward(self, states, actions):
        s = states
        a = actions
        x=torch.cat([s,a],1)
        x = self.seq(x)
        # x = F.relu(x)
        x = self.outer(x)
#         x = self.relu(x)

        return x
