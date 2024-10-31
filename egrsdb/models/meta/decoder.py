import math

import torch
import torch.nn.functional as F
from torch import nn

from egrsdb.models.meta.transformer_mlp_params import init_wb


class Conv1x1DecoderWithParams(nn.Module):
    def __init__(self, theta_dim, in_channels, hidden_channels, out_channels, depth):
        super().__init__()
        assert depth >= 3, "depth must be >= 3"
        # conv 1x1 kernel: in_channels, out_channels, 1, 1
        self.depth = depth
        self.theta_dim = theta_dim
        self.conv1x1_kernel = []
        self.theta_for_kernel = []
        for i in range(depth):
            if i == 0:
                in_to_hidden_shape = (in_channels, hidden_channels, 1, 1)
                self.conv1x1_kernel.append(nn.Parameter(init_wb(in_to_hidden_shape)))
                self.theta_for_kernel.append(
                    nn.Sequential(nn.LayerNorm(theta_dim), nn.Linear(theta_dim, hidden_channels))
                )
            elif i == depth - 1:
                hidden_to_out_shape = (hidden_channels, out_channels, 1, 1)
                self.conv1x1_kernel.append(nn.Parameter(init_wb(hidden_to_out_shape)))
                self.theta_for_kernel.append(nn.Sequential(nn.LayerNorm(theta_dim), nn.Linear(theta_dim, out_channels)))
            else:
                hidden_to_hidden_shape = (hidden_channels, hidden_channels, 1, 1)
                self.conv1x1_kernel.append(nn.Parameter(init_wb(hidden_to_hidden_shape)))
                self.theta_for_kernel.append(
                    nn.Sequential(nn.LayerNorm(theta_dim), nn.Linear(theta_dim, hidden_channels))
                )

    def update_params(self, theta):
        for i in range(self.depth):
            l, r = i * self.theta_dim, (i + 1) * self.theta_dim
            theta_shot = theta[:, l:r]
            x = self.theta_for_kernel[i](theta_shot)
            y = self.conv1x1_kernel[i]
            z = F.normalize(x * y, dim=1)
            self.conv1x1_kernel[i] = z

    def forward(self, coords, theta):
        # Calcueate
        x = coords
        for i in range(self.depth):
            if i == 0:
                x = F.gelu(F.conv2d(x, self.conv1x1_kernel[i], stride=1, padding=0))
            elif i == self.depth - 1:
                x = F.gelu(F.conv2d(x, self.conv1x1_kernel[i], stride=1, padding=0))
            else:
                x = F.gelu(F.conv2d(x, self.conv1x1_kernel[i], stride=1, padding=0))
        return x
