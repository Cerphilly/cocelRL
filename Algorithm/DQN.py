import torch
import torch.nn.functional as F
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight
from Network.Basic_Network import Policy_Network


class DQN:
    def __init__(self, state_dim, action_dim, device, args):
        self.device = device
        self.buffer = Buffer(state_dim=state_dim, action_dim=1, max_size=args.buffer_size, on_policy=False, device=self.device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.learning_rate = args.learning_rate
        self.epsilon = args.epsilon
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0
        self.copy_iter = args.copy_iter

        self.network = Policy_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_network = Policy_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)

        copy_weight(self.network, self.target_network)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.learning_rate)

        self.network_list = {'Network': self.network, 'Target_Network': self.target_network}
        self.name = 'DQN'

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(low=0, high=self.action_dim)
        else:
            with torch.no_grad():
                state = np.expand_dims(np.array(state), axis=0)
                state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                best_action = self.network(state, activation='linear').argmax(dim=1).item()
                return best_action

    def eval_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            best_action = self.network(state, activation='linear').argmax(dim=1).item()

        return best_action

    def train_network(self, s, a, r, ns, d):
        target_q, _ = self.target_network(ns, activation='linear').max(dim=1, keepdim=True)
        target_value = r + self.gamma * (1 - d) * target_q

        selected_value = self.network(s, activation='linear').gather(1, a.type(torch.int64))

        loss = F.smooth_l1_loss(input=selected_value, target=target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, training_num):
        total_loss = 0
        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            loss = self.train_network(s, a, r, ns, d)

            if self.current_step % self.copy_iter == 0:

                copy_weight(self.network, self.target_network)

            total_loss += loss

        return [['Loss/Loss', total_loss]]






