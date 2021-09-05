import torch
import torch.nn as nn

from Common.Utils import weight_init
from collections import OrderedDict

class Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-10, log_std_max=2):
        super(Gaussian_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.network = nn.Sequential(OrderedDict([('Layer1', nn.Linear(state_dim, hidden_dim)), ('ReLu1', nn.ReLU()),
                                                  ('Layer2', nn.Linear(hidden_dim, hidden_dim)), ('ReLu2', nn.ReLU()),
                                                  ('Layer3', nn.Linear(hidden_dim, action_dim * 2))]))

        self.apply(weight_init)


    def forward(self, state, deterministic=False):
        output = self.network(state)
        mu, log_std = output.chunk(2, dim=-1)



    def dist(self, state):
        pass

    def mu_sigma(self, state):
        pass

class Squashed_Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-10, log_std_max=2):
        super(Squashed_Gaussian_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.network = nn.Sequential(OrderedDict([('Layer1', nn.Linear(state_dim, hidden_dim)), ('ReLu1', nn.ReLU()),
                                                  ('Layer2', nn.Linear(hidden_dim, hidden_dim)), ('ReLu2', nn.ReLU()),
                                                  ('Layer3', nn.Linear(hidden_dim, action_dim * 2))]))

        self.apply(weight_init)

    def forward(self, state, deterministic=False):
        output = self.network(state)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std, validate_args=True)

        if deterministic == True:
            tanh_mean = torch.tanh(mean)
            log_prob = dist.log_prob(mean)

            log_pi = log_prob - torch.log(1 - tanh_mean.pow(2) + 1e-6).sum(dim=1, keepdim=True)
            return tanh_mean, log_pi

        else:
            sample_action = dist.rsample()
            tanh_sample = torch.tanh(sample_action)
            log_prob = dist.log_prob(sample_action)

            log_pi = log_prob - torch.log(1 - tanh_sample.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        return tanh_sample, log_pi



    def dist(self, state):
        output = self.network(state)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        return torch.distributions.Normal(mean, std)

    def mu_sigma(self, state):
        output = self.network(state)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        mean = torch.tanh(mean)

        return mean, std

    def entropy(self, state):
        dist = self.dist(state)
        return dist.entropy()

