import torch
import torch.nn as nn
from Common.Utils import weight_init
from collections import OrderedDict

class Policy_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, feature_dim = 50):
        super(Policy_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        if type(self.state_dim) == int:
            self.network = nn.Sequential(OrderedDict([('Layer1', nn.Linear(state_dim, hidden_dim)), ('ReLu1', nn.ReLU()),
                                                  ('Layer2', nn.Linear(hidden_dim, hidden_dim)), ('ReLu2', nn.ReLU()),
                                                  ('Layer3', nn.Linear(hidden_dim, action_dim))]))

        else:
            pass#encoder and network

        self.apply(weight_init)

    def forward(self, state, activation='tanh'):

        if type(self.state_dim) == int:
            output = self.network(state)

        if activation == 'tanh':
            output = torch.tanh(output)
        elif activation == 'softmax':
            output = torch.softmax(output, dim=-1)

        return output


class Q_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, feature_dim=50):
        super(Q_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        if type(self.state_dim) == int:
            self.network = nn.Sequential(OrderedDict([('Layer1', nn.Linear(state_dim + action_dim, hidden_dim)), ('ReLu1', nn.ReLU()),
                                                  ('Layer2', nn.Linear(hidden_dim, hidden_dim)), ('ReLu2', nn.ReLU()),
                                                  ('Layer3', nn.Linear(hidden_dim, 1))]))

        else:
            pass

        self.apply(weight_init)

    def forward(self, state, action):
        if type(self.state_dim) == int:
            output = self.network(torch.cat([state, action], dim=-1))
        else:
            pass

        return output


class V_Network(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, feature_dim=50):
        super(V_Network, self).__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim

        if type(self.state_dim) == int:
            self.network = nn.Sequential(OrderedDict([('Layer1', nn.Linear(state_dim, hidden_dim)), ('ReLu1', nn.ReLU()),
                                                  ('Layer2', nn.Linear(hidden_dim, hidden_dim)), ('ReLu2', nn.ReLU()),
                                                  ('Layer3', nn.Linear(hidden_dim, 1))]))

        else:
            pass

        self.apply(weight_init)

    def forward(self, state):

        if type(self.state_dim) == int:
            output = self.network(state)

        else:
            pass

        return output