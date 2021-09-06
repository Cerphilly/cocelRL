#Proximal Policy Optimization Algorithms, Schulman et al, 2017
#High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
#https://spinningup.openai.com/en/latest/algorithms/ppo.html

import torch
import torch.nn.functional as F
import numpy as np

from Common.Buffer import Buffer
from Network.Basic_Network import Policy_Network, V_Network
from Network.Gaussian_Actor import Gaussian_Actor

class PPO:
    def __init__(self, state_dim, action_dim, device, args):
        self.device = device
        self.discrete = args.discrete

        self.buffer = Buffer(state_dim, action_dim if args.discrete == False else 1, args.buffer_size, on_policy=True, device=self.device)

        self.ppo_mode = args.ppo_mode  # mode: 'clip'
        assert self.ppo_mode is 'clip'

        self.gamma = args.gamma
        self.lambda_gae = args.lambda_gae
        self.batch_size = args.batch_size
        self.clip = args.clip

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_start = 0
        self.training_step = args.training_step

        if self.discrete == True:
            self.actor = Policy_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        else:
            self.actor = Gaussian_Actor(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)

        self.critic = V_Network(self.state_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}
        self.name = 'PPO'

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


    def train_actor(self, batch_s, batch_a, batch_old_log_policy, batch_advantages):

        if self.discrete == True:
            policy = self.actor(batch_s, activation='softmax')
            dist = torch.distributions.Categorical(probs=policy)
            log_policy = dist.log_prob(batch_a.squeeze()).reshape(-1, 1)
            ratio = (log_policy - batch_old_log_policy).exp()
            surrogate = ratio * batch_advantages
        else:
            dist = self.actor.dist(batch_s)
            log_policy = dist.log_prob(batch_a)
            ratio = (log_policy - batch_old_log_policy).exp()
            surrogate = ratio * batch_advantages

        if self.ppo_mode == 'clip':
            clipped_surrogate = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages
            actor_loss = -torch.minimum(surrogate, clipped_surrogate)
            actor_loss = actor_loss.mean()

        else:
            raise NotImplementedError

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def train_critic(self, batch_s, batch_returns):
        critic_loss = F.mse_loss(input=self.critic(batch_s), target=batch_returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def train(self, training_num):
        total_a_loss, total_c_loss = 0, 0

        s, a, r, ns, d, log_prob = self.buffer.all_sample()

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

        advantages = (advantages - advantages.mean().item()) / (advantages.std(dim=0).item())
        advantages = advantages.detach()

        n = len(s)
        arr = np.arange(n)
        training_num2 = max(int(n / self.batch_size), 1)

        for i in range(training_num):
            for epoch in range(training_num2):
                if epoch < training_num2 - 1:
                    batch_index = arr[self.batch_size * epoch:  self.batch_size * (epoch + 1)]
                else:
                    batch_index = arr[self.batch_size * epoch: ]

                batch_s = s[batch_index]
                batch_a = a[batch_index]
                batch_returns = returns[batch_index]
                batch_advantages = advantages[batch_index]
                batch_old_log_policy = log_prob[batch_index]

                total_a_loss += self.train_actor(batch_s, batch_a, batch_old_log_policy, batch_advantages)
                total_c_loss += self.train_critic(batch_s, batch_returns)


        self.buffer.delete()
        return [['Loss/Actor', total_a_loss], ['Loss/Critic', total_c_loss]]



