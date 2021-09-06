import torch
import torch.nn.functional as F
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update
from Network.Basic_Network import Policy_Network, Q_Network



class DDPG:
    def __init__(self, state_dim, action_dim, device, args):
        self.device = device
        self.buffer = Buffer(state_dim=state_dim, action_dim=action_dim, max_size=args.buffer_size, on_policy=False, device=self.device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.noise_scale = args.noise_scale
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0

        self.actor = Policy_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_actor = Policy_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.critic = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic, self.target_critic)

        self.network_list = {'Actor': self.actor, 'Target_Actor': self.target_actor, 'Critic': self.critic,
                             'Target_Critic': self.target_critic}
        self.name = 'DDPG'


    def get_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)

            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            noise = np.random.normal(loc=0, scale=self.noise_scale, size=self.action_dim)
            action = self.actor(state).cpu().numpy()[0] + noise

            action = np.clip(action, -1, 1)

        return action

    def eval_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)

            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            action = self.actor(state).cpu().numpy()[0]
            action = np.clip(action, -1, 1)

        return action

    def train_actor(self, s):
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def train_critic(self, s, a, r, ns, d):
        value_next = self.target_critic(ns, self.target_actor(ns))
        target_value = r + (1 - d) * self.gamma * value_next

        critic_loss = F.mse_loss(input=self.critic(s, a), target=target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def train(self, training_num):
        total_a_loss = 0
        total_c_loss = 0

        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            total_a_loss += self.train_actor(s.detach())
            total_c_loss += self.train_critic(s.detach(), a.detach(), r.detach(), ns.detach(), d.detach())

            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic, self.target_critic, self.tau)

        return [['Loss/Actor', total_a_loss], ['Loss/Critic', total_c_loss]]





