#Soft Actor-Critic Algorithms and Applications, Haarnoja et al, 2018

import torch
import torch.nn.functional as F
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update
from Network.Basic_Network import Q_Network
from Network.Gaussian_Actor import Squashed_Gaussian_Actor
from Network.Encoder import PixelEncoder

class ImageSAC_v2:
    def __init__(self, obs_dim, action_dim, device, args):
        self.device = device
        self.buffer = Buffer(state_dim=obs_dim, action_dim=action_dim, max_size=args.buffer_size, on_policy=False, device=self.device)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0
        self.critic_update = args.critic_update

        self.feature_dim = args.feature_dim
        self.layer_num = args.layer_num
        self.filter_num = args.filter_num
        self.tau = args.tau
        self.encoder_tau = args.encoder_tau

        self.target_entropy = -action_dim
        self.log_alpha = torch.as_tensor(np.log(args.alpha), dtype=torch.float32, device=self.device).requires_grad_()
        self.optimize_alpha = args.train_alpha

        self.actor = Squashed_Gaussian_Actor(self.feature_dim, self.action_dim, args.hidden_dim, args.log_std_min, args.log_std_max).to(self.device)
        self.critic1 = Q_Network(self.feature_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.critic2 = Q_Network(self.feature_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic1 = Q_Network(self.feature_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic2 = Q_Network(self.feature_dim, self.action_dim, args.hidden_dim).to(self.device)

        self.encoder = PixelEncoder(self.obs_dim, self.feature_dim, self.layer_num, self.filter_num).to(self.device)
        self.target_encoder = PixelEncoder(self.obs_dim, self.feature_dim, self.layer_num, self.filter_num).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2, 'Encoder': self.encoder, 'Target_Encoder': self.target_encoder}
        self.name = 'ImageSAC_v2'

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()


    def get_action(self, observation):
        with torch.no_grad():
            observation = np.expand_dims(np.array(observation), axis=0)
            observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            feature = self.encoder(observation)
            action, _ = self.actor(feature)

            action = np.clip(action.cpu().numpy()[0], -1, 1)

        return action

    def eval_action(self, observation):
        with torch.no_grad():
            observation = np.expand_dims(np.array(observation), axis=0)
            observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            feature = self.encoder(observation)
            action, _ = self.actor(feature, deterministic=True)

            action = np.clip(action.cpu().numpy()[0], -1, 1)

        return action

    def train_alpha(self, s):
        _, s_logpi = self.actor(self.encoder(s).detach())
        alpha_loss = -(self.log_alpha.exp() * (s_logpi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def train_critic(self, s, a, r, ns, d):
        with torch.no_grad():
            ns_action, ns_logpi = self.actor(self.encoder(ns))
            target_min_aq = torch.minimum(self.target_critic1(self.target_encoder(ns), ns_action), self.target_critic2(self.target_encoder(ns), ns_action))
            target_q = (r + self.gamma * (1 - d) * (target_min_aq - self.alpha * ns_logpi))

        critic1_loss = F.mse_loss(input=self.critic1(self.encoder(s), a), target=target_q)
        critic2_loss = F.mse_loss(input=self.critic2(self.encoder(s), a), target=target_q)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        critic1_loss.backward(retain_graph=True)
        critic2_loss.backward()

        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        self.encoder_optimizer.step()

        return (critic1_loss.item(), critic2_loss.item())

    def train_actor(self, s):
        s_action, s_logpi = self.actor(self.encoder(s).detach())
        min_aq_rep = torch.minimum(self.critic1(self.encoder(s).detach(), s_action), self.critic2(self.encoder(s).detach(), s_action))
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
                soft_update(self.encoder, self.target_encoder, self.encoder_tau)


        return [['Loss/Actor', total_a_loss], ['Loss/Critic1', total_c1_loss], ['Loss/Critic2', total_c2_loss], ['Loss/alpha', total_alpha_loss], ['Alpha', self.alpha]]


