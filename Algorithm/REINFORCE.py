#Simple statistical gradient-following algorithms for connectionist reinforcement learning, Ronald J. Williams, 1992

import torch
import torch.nn.functional as F
import numpy as np

from Common.Buffer import Buffer
from Network.Basic_Network import Policy_Network
from Network.Gaussian_Actor import Gaussian_Actor


class REINFORCE:
    def __init__(self, state_dim, action_dim, device, args):
        self.device = device
        self.buffer = Buffer(state_dim=state_dim, action_dim=action_dim if args.discrete == False else 1, max_size=args.buffer_size, on_policy=True, device=self.device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.discrete = args.discrete

        self.gamma = args.gamma
        self.training_start = 0
        self.training_step = 1

        if args.discrete == True:
            self.network = Policy_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        else:
            self.network = Gaussian_Actor(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)

        self.network_optimizer = torch.optim.Adam(self.network.parameters(), lr=args.learning_rate)

        self.network_list = {'Network': self.network}
        self.name = 'REINFORCE'


    def get_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            if self.discrete == True:
                policy = self.network(state, activation='softmax')
                dist = torch.distributions.Categorical(probs=policy)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]

            else:
                action, log_prob = self.network(state)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]

        return action, log_prob

    def eval_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            if self.discrete == True:
                policy = self.network(state, activation='softmax')
                action = torch.argmax(policy, dim=1).cpu().numpy()[0]

            else:
                action, _ = self.network(state, deterministic=True)
                action = action.cpu().numpy()[0]

            return action


    def train_network(self, s, a, returns):
        if self.discrete == True:
            policy = self.network(s, activation='softmax')
            dist = torch.distributions.Categorical(probs=policy)
            log_policy = dist.log_prob(a.squeeze()).reshape((-1, 1))

        else:
            mu, sigma = self.network.mu_sigma(s)
            dist = torch.distributions.Normal(mu, sigma, validate_args=True)
            log_policy = dist.log_prob(a)

        loss = (-log_policy * returns).sum()
        self.network_optimizer.zero_grad()
        loss.backward()
        self.network_optimizer.step()

        return loss.item()


    def train(self, training_num):
        total_loss = 0
        s, a, r, ns, d, _ = self.buffer.all_sample()

        returns = np.zeros_like(r.cpu().numpy())

        running_return = 0
        for t in reversed(range(len(r))):
            running_return = r[t] + self.gamma * running_return * (1 - d[t])
            returns[t] = running_return

        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        total_loss += self.train_network(s, a, returns)

        self.buffer.delete()
        return [['Loss/Loss', total_loss]]




