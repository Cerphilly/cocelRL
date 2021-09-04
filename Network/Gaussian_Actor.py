import torch
import torch.nn as nn
from Common.Utils import weight_init
from collections import OrderedDict

class Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-10, log_std_max=2):
        super(Gaussian_Actor, self).__init__()

    def forward(self, state, deterministic=False):
        pass

    def dist(self, state):
        pass

    def mu_sigma(self, state):
        pass

class Squashed_Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-10, log_std_max=2):
        super(Squashed_Gaussian_Actor, self).__init__()


    def forward(self, state, deterministic=False):
        pass

    def dist(self, state):
        pass

    def mu_sigma(self, state):
        pass

    def entropy(self, state):
        pass

