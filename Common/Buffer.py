import numpy as np
from Common.Utils import random_crop, center_crop_images
import torch

class Buffer:
    def __init__(self, state_dim, action_dim, max_size=1e6, on_policy=False, device=None):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.on_policy = on_policy

        self.device = device
        if self.device is None:
            assert ValueError

        if type(self.state_dim) == int:
            self.s = np.empty((self.max_size, self.state_dim), dtype=np.float32)
            self.ns = np.empty((self.max_size, self.state_dim), dtype=np.float32)
        else:
            self.s = np.empty((self.max_size, *self.state_dim), dtype=np.uint8)
            self.ns = np.empty((self.max_size, *self.state_dim), dtype=np.uint8)

        self.a = np.empty((self.max_size, self.action_dim), dtype=np.float32)
        self.r = np.empty((self.max_size, 1), dtype=np.float32)
        self.d = np.empty((self.max_size, 1), dtype=np.float32)

        if self.on_policy == True:
            self.log_prob = np.empty((self.max_size, self.action_dim), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        if self.full == False:
            return self.idx
        else:
            return self.max_size

    def add(self, s, a, r, ns, d, log_prob=None):
        np.copyto(self.s[self.idx], s)
        np.copyto(self.a[self.idx], a)
        np.copyto(self.r[self.idx], r)
        np.copyto(self.ns[self.idx], ns)
        np.copyto(self.d[self.idx], d)

        if self.on_policy == True:
            np.copyto(self.log_prob[self.idx], log_prob)

        self.idx = (self.idx + 1) % self.max_size
        if self.idx == 0:
            self.full = True

    def delete(self):
        if type(self.state_dim) == int:
            self.s = np.empty((self.max_size, self.state_dim), dtype=np.float32)
            self.ns = np.empty((self.max_size, self.state_dim), dtype=np.float32)
        else:
            self.s = np.empty((self.max_size, *self.state_dim), dtype=np.uint8)
            self.ns = np.empty((self.max_size, *self.state_dim), dtype=np.uint8)

        self.a = np.empty((self.max_size, self.action_dim), dtype=np.float32)
        self.r = np.empty((self.max_size, 1), dtype=np.float32)
        self.d = np.empty((self.max_size, 1), dtype=np.float32)

        if self.on_policy == True:
            self.log_prob = np.empty((self.max_size, self.action_dim), dtype=np.float32)

        self.idx = 0
        self.full = False

    def all_sample(self):
        ids = np.arange(self.max_size if self.full else self.idx)
        states = torch.as_tensor(self.s[ids], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.a[ids], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self.r[ids], dtype=torch.float32, device=self.device)
        states_next = torch.as_tensor(self.ns[ids], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.d[ids], dtype=torch.float32, device=self.device)

        if self.on_policy == True:
            log_probs = torch.as_tensor(self.log_prob[ids], dtype=torch.float32, device=self.device)
            return states, actions, rewards, states_next, dones, log_probs

        return states, actions, rewards, states_next, dones

    def sample(self, batch_size):
        ids = np.random.randint(0, self.max_size if self.full else self.idx, size=batch_size)
        states = torch.as_tensor(self.s[ids], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.a[ids], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self.r[ids], dtype=torch.float32, device=self.device)
        states_next = torch.as_tensor(self.ns[ids], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.d[ids], dtype=torch.float32, device=self.device)

        if self.on_policy == True:
            log_probs = torch.as_tensor(self.log_prob[ids], dtype=torch.float32, device=self.device)
            return states, actions, rewards, states_next, dones, log_probs

        return states, actions, rewards, states_next, dones

    def cpc_sample(self, batch_size, image_size=84):
        # ImageRL/CURL
        ids = np.random.randint(0, self.max_size if self.full else self.idx, size=batch_size)

        states = self.s[ids]
        actions = self.a[ids]
        rewards = self.r[ids]
        states_next = self.ns[ids]
        dones = self.d[ids]

        pos = states.copy()

        states = random_crop(states, image_size)
        states_next = random_crop(states_next, image_size)
        pos = random_crop(pos, image_size)

        cpc_kwargs = dict(obs_anchor=states, obs_pos=pos, time_anchor=None, time_pos=None)

        if self.on_policy == True:
            log_probs = self.log_prob[ids]
            return states, actions, rewards, states_next, dones, log_probs, cpc_kwargs

        return states, actions, rewards, states_next, dones, cpc_kwargs

    def rad_sample(self, batch_size, aug_funcs, pre_image_size=100):

        ids = np.random.randint(0, self.max_size if self.full else self.idx, size=batch_size)

        states = self.s[ids]
        states_next = self.ns[ids]


        for aug, func in aug_funcs.items():
            if 'crop' in aug or 'cutout' in aug:
                states = func(states)
                states_next = func(states_next)

            elif 'translate' in aug:
                states = center_crop_images(states, pre_image_size)
                states_next = center_crop_images(states_next, pre_image_size)

                states, random_idxs = func(states, return_random_idxs=True)
                states_next = func(states_next, **random_idxs)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.a[ids], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self.r[ids], dtype=torch.float32, device=self.device)
        states_next = torch.as_tensor(states_next, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.d[ids], dtype=torch.float32, device=self.device)

        states = states / 255.
        states_next = states_next / 255.

        for aug, func in aug_funcs.items():
            if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                continue
            states = func(states)
            states_next = func(states_next)

        if self.on_policy == True:
            log_probs = torch.as_tensor(self.log_prob[ids], dtype=torch.float32, device=self.device)
            return states, actions, rewards, states_next, dones, log_probs

        return states, actions, rewards, states_next, dones













