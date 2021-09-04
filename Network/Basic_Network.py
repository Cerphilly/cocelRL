import torch
import torch.nn as nn
from Common.Utils import weight_init
from collections import OrderedDict

class Policy_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Policy_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(OrderedDict([('Layer1', nn.Linear(state_dim, hidden_dim)), ('ReLu1', nn.ReLU()),
                                                  ('Layer2', nn.Linear(hidden_dim, hidden_dim)), ('ReLu2', nn.ReLU()),
                                                  ('Layer3', nn.Linear(hidden_dim, action_dim))]))

        self.apply(weight_init)

    def forward(self, state, activation='tanh'):
        output = self.network(state)

        if activation == 'tanh':
            output = torch.tanh(output)
        elif activation == 'softmax':
            output = torch.softmax(output, dim=-1)

        return output


class Q_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Q_Network, self).__init__()

        self.network = nn.Sequential(OrderedDict([('Layer1', nn.Linear(state_dim + action_dim, hidden_dim)), ('ReLu1', nn.ReLU()),
                                                  ('Layer2', nn.Linear(hidden_dim, hidden_dim)), ('ReLu2', nn.ReLU()),
                                                  ('Layer3', nn.Linear(hidden_dim, 1))]))

        self.apply(weight_init)

    def forward(self, state, action):
        output = self.network(torch.cat([state, action], dim=-1))

        return output


class V_Network(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(V_Network, self).__init__()

        self.network = nn.Sequential(OrderedDict([('Layer1', nn.Linear(state_dim, hidden_dim)), ('ReLu1', nn.ReLU()),
                                                  ('Layer2', nn.Linear(hidden_dim, hidden_dim)), ('ReLu2', nn.ReLU()),
                                                  ('Layer3', nn.Linear(hidden_dim, 1))]))
        self.apply(weight_init)

    def forward(self, state):
        output = self.network(state)

        return output