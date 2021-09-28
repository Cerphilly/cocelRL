import torch
import torch.nn as nn
import torch.nn.functional as F
from Common.Utils import weight_init

OUT_DIM_64 = {2: 29, 4: 25, 6: 21}
OUT_DIM_84 = {2: 39, 4: 35, 6: 31}
OUT_DIM_108 = {2: 51, 4: 47, 6: 43}

class PixelEncoder(nn.Module):
    def __init__(self, obs_dim, feature_dim, layer_num=4, filter_num=32):
        super(PixelEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.layer_num = layer_num
        self.filter_num = filter_num

        self.conv = nn.ModuleList([nn.Conv2d(in_channels=obs_dim[0], out_channels=filter_num, kernel_size=(3, 3), stride=(2, 2))])
        self.conv.extend([nn.Conv2d(in_channels=filter_num, out_channels=filter_num, kernel_size=(3, 3), stride=(1, 1)) for _ in range(layer_num - 1)])

        if obs_dim[-1] == 64:
            fc_dim = OUT_DIM_64[layer_num]
        elif obs_dim[-1] == 84:
            fc_dim = OUT_DIM_84[layer_num]
        elif obs_dim[-1] == 108:
            fc_dim = OUT_DIM_108[layer_num]
        else:
            raise ValueError

        self.fc = nn.Linear(filter_num * fc_dim * fc_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.apply(weight_init)

    def forward(self, observation, activation='tanh'):
        z = observation / 255.

        for i in range(len(self.conv)):
            z = F.relu(self.conv[i](z))

        z = z.view(z.size()[0], -1)
        z = self.fc(z)
        z = self.ln(z)

        if activation == 'tanh':
            z = torch.tanh(z)

        return z






if __name__ == '__main__':
    a = PixelEncoder((3, 84, 84), 50, layer_num=4)
    # import numpy as np
    b = torch.zeros((1, 3, 84, 84))
    from Network.Basic_Network import Policy_Network
    c = Policy_Network(50, 3, encoder=a)
    print(c)
    print(c.parameters())
    for i in c.parameters():
        print(i.shape)
    print("----------------")
    for i in c.encoder.parameters():
        print(i.shape)

