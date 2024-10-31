#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/3 08:58
import numpy as np
import torch
from torch import nn


def backward_warp(image, flow):
    """
    :param image: [B, C, H, W]
    :param flow: [B, 2, H, W]
    :return:
    """
    B, C, H, W = image.size()
    # generate grid
    grid_horizontal = torch.linspace(-1.0 + (1.0 / W), 1.0 - (1.0 / W), W).view(1, 1, 1, -1).expand(-1, -1, H, -1)
    grid_vertical = torch.linspace(-1.0 + (1.0 / H), 1.0 - (1.0 / H), H).view(1, 1, -1, 1).expand(-1, -1, -1, W)
    grid = torch.cat([grid_horizontal, grid_vertical], dim=1).cuda()
    # Normalized flow
    normal_flow_horizontal = flow[:, 0:1, :, :] / (W - 1.0) * 2.0
    normal_flow_vertical = flow[:, 1:2, :, :] / (H - 1.0) * 2.0
    normal_flow = torch.cat(
        [normal_flow_horizontal, normal_flow_vertical],
        1,
    )
    # Warp
    warped_grid = grid + normal_flow
    # B, 2, H, W -> B, H, W, 2. v \in warped_grid, v = [y, x]
    warped_grid = warped_grid.permute(0, 2, 3, 1)
    warped_image = torch.nn.functional.grid_sample(image, warped_grid)
    return warped_image


class BackwardWarp(nn.Module):
    """
    A class for creating a backwarping object.
    This is used for backwarping to an image:
    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1,
    it generates I0 <-- backwarp(F_0_1, I1).
    Returns output tensor after passing input `img` and `flow` to the backwarping block.
    """

    def __init__(self, W: int, H: int, device):
        """
        :param W: width of the image.
        :param H: height of the image.
        """
        super(BackwardWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block. I0  = backwarp(I1, F_0_1)
        :param img: tensor frame I1.
        :param flow: tensor optical flow from I0 and I1: F_0_1.
        :return: I0
        """
        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2 * (x / self.W - 0.5)
        y = 2 * (y / self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        img_out = torch.nn.functional.grid_sample(img, grid)
        return img_out
