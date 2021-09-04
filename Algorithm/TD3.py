import torch
import torch.nn.functional as F
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update
from Network.Basic_Network import Policy_Network, Q_Network

class TD3:
    def __init__(self, state_dim, action_dim, device, args):
        self.device = device
        self.buffer = Buffer(state_dim=state_dim, action_dim=action_dim, max_size=args.buffer_size, on_policy=False, device=self.device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.policy_delay = args.policy_delay
        self.actor_noise = args.actor_noise
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0

        self.actor = Policy_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_actor = Policy_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.critic1 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic1 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.critic2 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic2 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.critic_lr)

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2}
        self.name = 'TD3'

    def get_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            noise = np.random.normal(loc=0, scale=self.actor_noise, size=self.action_dim)
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
        actor_loss = -self.critic1(s, self.actor(s)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def train_critic(self, s, a, r, ns, d):
        target_action = self.target_actor(ns)
        target_action = torch.clamp(target_action + torch.clamp(torch.normal(mean=0, std=self.target_noise, size=target_action.shape).to(self.device), -self.noise_clip, self.noise_clip), -1, 1)

        target_value = r + self.gamma * (1 - d) * torch.minimum(self.target_critic1(ns, target_action), self.target_critic2(ns, target_action))

        critic1_loss = F.mse_loss(input=self.critic1(s, a), target=target_value)
        critic2_loss = F.mse_loss(input=self.critic2(s, a), target=target_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return (critic1_loss.item(), critic2_loss.item())

    def train(self, training_num):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0

        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            critic1_loss, critic2_loss = self.train_critic(s, a, r, ns, d)
            total_c1_loss += critic1_loss
            total_c2_loss += critic2_loss

            if self.current_step % self.policy_delay == 0:
                total_a_loss += self.train_actor(s)

                soft_update(self.actor, self.target_actor, self.tau)
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)

        return [['Loss/Actor', total_a_loss], ['Loss/Critic1', total_c1_loss], ['Loss/Critic2', total_c2_loss]]





