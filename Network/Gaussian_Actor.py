import torch
import torch.nn as nn
import torch.nn.functional as F

from Common.Utils import weight_init
from collections import OrderedDict

class Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = (256, 256), log_std_min=-10, log_std_max=2):
        super(Gaussian_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.network = nn.ModuleList([nn.Linear(state_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], action_dim * 2))

        self.apply(weight_init)


    def forward(self, state, deterministic=False):
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)

        mean, log_std = z.chunk(2, dim=-1)
        mean = torch.tanh(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std, validate_args=True)

        if deterministic == True:
            log_prob = dist.log_prob(mean)
            return mean, log_prob

        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action, log_prob

    def dist(self, state):
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)

        mean, log_std = z.chunk(2, dim=-1)
        mean = torch.tanh(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std, validate_args=True)

        return dist

    def mu_sigma(self, state):
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)

        mean, log_std = z.chunk(2, dim=-1)
        mean = torch.tanh(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        return mean, std

class Squashed_Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256), log_std_min=-10, log_std_max=2):
        super(Squashed_Gaussian_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.network = nn.ModuleList([nn.Linear(state_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], action_dim * 2))

        self.apply(weight_init)

    def forward(self, state, deterministic=False):
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)

        mean, log_std = z.chunk(2, dim=-1)
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
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)

        mean, log_std = z.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        return torch.distributions.Normal(mean, std)

    def mu_sigma(self, state):
        z = state
        for i in range(len(self.network)):
            z = self.network[i](z)

        mean, log_std = z.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        mean = torch.tanh(mean)

        return mean, std

    def entropy(self, state):
        dist = self.dist(state)
        return dist.entropy()



if __name__ == '__main__':
    a = Squashed_Gaussian_Actor(3, 2)
    print(a)

