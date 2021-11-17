import torch
import torch.nn.functional as F
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update
from Network.Basic_Network import Q_Network
from Network.Gaussian_Actor import Squashed_Gaussian_Actor

class SAC_v2:
    def __init__(self, state_dim, action_dim, device, args):
        self.device = device
        self.buffer = Buffer(state_dim=state_dim, action_dim=action_dim, max_size=args.buffer_size, on_policy=False, device=self.device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0
        self.critic_update = args.critic_update

        self.target_entropy = -action_dim
        self.log_alpha = torch.as_tensor(np.log(args.alpha), dtype=torch.float32, device=self.device).requires_grad_()
        self.optimize_alpha = args.train_alpha

        self.actor = Squashed_Gaussian_Actor(self.state_dim, self.action_dim, args.hidden_dim, args.log_std_min, args.log_std_max).to(self.device)
        self.critic1 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.critic2 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic1 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic2 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2}
        self.name = 'SAC_v2'

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def get_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, _ = self.actor(state)

            action = np.clip(action.cpu().numpy()[0], -1, 1)

        return action

    def eval_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, _ = self.actor(state, deterministic=True)

            action = np.clip(action.cpu().numpy()[0], -1, 1)

        return action

    def train_alpha(self, s):
        _, s_logpi = self.actor(s)
        alpha_loss = -(self.log_alpha.exp() * (s_logpi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def train_critic(self, s, a, r, ns, d):
        with torch.no_grad():
            ns_action, ns_logpi = self.actor(ns)
            target_min_aq = torch.min(self.target_critic1(ns, ns_action), self.target_critic2(ns, ns_action))
            target_q = (r + self.gamma * (1 - d) * (target_min_aq - self.alpha * ns_logpi)).detach()

        critic1_loss = F.mse_loss(input=self.critic1(s, a), target=target_q)
        critic2_loss = F.mse_loss(input=self.critic2(s, a), target=target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return (critic1_loss.item(), critic2_loss.item())

    def train_actor(self, s):
        s_action, s_logpi = self.actor(s)
        min_aq_rep = torch.min(self.critic1(s, s_action), self.critic2(s, s_action))
        actor_loss = (self.alpha * s_logpi - min_aq_rep).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def train(self, training_num):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_alpha_loss = 0
        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            critic1_loss, critic2_loss = self.train_critic(s, a, r, ns, d)
            total_c1_loss += critic1_loss
            total_c2_loss += critic2_loss
            total_a_loss += self.train_actor(s)

            if self.optimize_alpha == True:
                total_alpha_loss += self.train_alpha(s)

            if self.current_step % self.critic_update == 0:
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)


        return [['Loss/Actor', total_a_loss], ['Loss/Critic1', total_c1_loss], ['Loss/Critic2', total_c2_loss], ['Loss/alpha', total_alpha_loss], ['Alpha', self.alpha]]

