#Policy Gradient Methods for Reinforcement Learning with Function Approximation, Sutton et al, 2000
#High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
#https://spinningup.openai.com/en/latest/algorithms/vpg.html

import torch
import torch.nn.functional as F
import numpy as np

from Common.Buffer import Buffer
from Network.Basic_Network import Policy_Network, V_Network
from Network.Gaussian_Actor import Gaussian_Actor


class VPG:
    def __init__(self, state_dim, action_dim, device, args):
        self.device = device
        self.buffer = Buffer(state_dim, action_dim if args.discrete == False else 1, args.buffer_size, on_policy=True, device=self.device)
        self.discrete = args.discrete

        self.gamma = args.gamma
        self.lambda_gae = args.lambda_gae

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_start = 0
        self.training_step = 1

        if self.discrete == True:
            self.actor = Policy_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        else:
            self.actor = Gaussian_Actor(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)

        self.critic = V_Network(self.state_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}
        self.name = 'VPG'

    def get_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            if self.discrete == True:
                policy = self.actor(state, activation='softmax')
                dist = torch.distributions.Categorical(probs=policy)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]

            else:
                action, log_prob = self.actor(state)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]

        return action, log_prob

    def eval_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            if self.discrete == True:
                policy = self.actor(state, activation='softmax')
                action = torch.argmax(policy, dim=1).cpu().numpy()[0]

            else:
                action, _ = self.actor(state, deterministic=True)
                action = action.cpu().numpy()[0]

        return action

    def train_actor(self, s, a, advantages):
        if self.discrete == True:
            policy = self.actor(s, activation='softmax')
            dist = torch.distributions.Categorical(probs=policy)
            log_policy = dist.log_prob(a.squeeze()).reshape((-1, 1))

        else:
            dist = self.actor.dist(s)
            log_policy = dist.log_prob(a)

        actor_loss = - log_policy * advantages
        actor_loss = actor_loss.sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()


    def train_critic(self, s, returns):
        critic_loss = F.mse_loss(input=self.critic(s), target=returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()


    def train(self, training_num):
        total_a_loss, total_c_loss = 0, 0

        s, a, r, ns, d, _ = self.buffer.all_sample()
        values = self.critic(s)

        returns = torch.zeros_like(r)
        advantages = torch.zeros_like(returns)
        running_return = torch.zeros(1, device=self.device)
        previous_value = torch.zeros(1, device=self.device)
        running_advantage = torch.zeros(1, device=self.device)

        for t in reversed(range(len(r))):
            running_return = (r[t] + self.gamma * running_return * (1 - d[t]))
            running_tderror = (r[t] + self.gamma * previous_value * (1 - d[t]) - values[t])
            running_advantage = (running_tderror + (self.gamma * self.lambda_gae) * running_advantage * (1 - d[t]))

            returns[t] = running_return
            previous_value = values[t]
            advantages[t] = running_advantage

        total_a_loss += self.train_actor(s, a, advantages.detach())
        total_c_loss += self.train_critic(s, returns.detach())

        self.buffer.delete()
        return [['Loss/Actor', total_a_loss], ['Loss/Critic', total_c_loss]]








