import torch
import torch.nn as nn
import torch.nn.functional as F
from Common.Utils import weight_init
from collections import OrderedDict

class Policy_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256), feature_dim = 50):
        super(Policy_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim


        self.network = nn.ModuleList([nn.Linear(state_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], action_dim))

        self.apply(weight_init)

    def forward(self, state, activation='tanh'):
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)

        if activation == 'tanh':
            output = torch.tanh(z)
        elif activation == 'softmax':
            output = torch.softmax(z, dim=-1)
        else:
            output = z

        return output


class Q_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256), feature_dim=50):
        super(Q_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim


        self.network = nn.ModuleList([nn.Linear(state_dim + action_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], 1))

        self.apply(weight_init)

    def forward(self, state, action):
        z = torch.cat([state, action], dim=-1)
        for i in range(len(self.network)):
            z = self.network[i](z)
        else:
            pass

        return z


class V_Network(nn.Module):
    def __init__(self, state_dim, hidden_dim=(256, 256), feature_dim=50):
        super(V_Network, self).__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim

        self.network = nn.ModuleList([nn.Linear(state_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], 1))

        self.apply(weight_init)

    def forward(self, state):
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)


        return z


if __name__ == '__main__':
    a = V_Network(3)
    print(a)