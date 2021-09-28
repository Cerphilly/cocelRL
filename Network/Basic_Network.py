import torch
import torch.nn as nn
import torch.nn.functional as F
from Common.Utils import weight_init
from Network.Encoder import PixelEncoder

class Policy_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256), encoder=None):
        super(Policy_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.encoder = encoder

        if self.encoder is not None:
            assert self.encoder.feature_dim == state_dim

        self.network = nn.ModuleList([nn.Linear(state_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], action_dim))


        self.apply(weight_init)

    def forward(self, state, activation='tanh'):
        if self.encoder is None:
            z = state
        else:
            z = self.encoder(state)

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
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256), encoder=None):
        super(Q_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.encoder = encoder

        if self.encoder is not None:
            assert self.encoder.feature_dim == state_dim

        self.network = nn.ModuleList([nn.Linear(state_dim + action_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], 1))

        self.apply(weight_init)

    def forward(self, state, action):
        if self.encoder is None:
            z = torch.cat([state, action], dim=-1)
        else:
            state = self.encoder(state)
            z = torch.cat([state, action], dim=-1)

        for i in range(len(self.network)):
            z = self.network[i](z)

        return z


class V_Network(nn.Module):
    def __init__(self, state_dim, hidden_dim=(256, 256), encoder=None):
        super(V_Network, self).__init__()
        self.state_dim = state_dim

        self.encoder = encoder

        if self.encoder is not None:
            assert self.encoder.feature_dim == state_dim

        self.network = nn.ModuleList([nn.Linear(state_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], 1))
        self.encoder = encoder

        self.apply(weight_init)

    def forward(self, state):
        if self.encoder is None:
            z = state
        else:
            z = self.encoder(state)

        for i in range(len(self.network)):
            z = self.network[i](z)


        return z


if __name__ == '__main__':
    a = V_Network(3)
    print(a)