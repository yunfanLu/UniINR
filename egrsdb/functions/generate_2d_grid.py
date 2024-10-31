#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/2 16:21
import torch


def generate_2d_grid_WH(H, W):
    """
    generate 2d grid
    :param H: the high of the grid
    :param W: the width of the grid
    :return: [2, H, W]
    """
    x = torch.arange(0, W, 1).float()
    y = torch.arange(0, H, 1).float()
    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    grid = torch.stack([xx, yy], dim=0)
    return grid


def generate_2d_grid_HW(H, W):
    """
    generate 2d grid
    :param H: the high of the grid
    :param W: the width of the grid
    :return: [2, H, W]
    """
    x = torch.arange(0, H, 1).float()
    y = torch.arange(0, W, 1).float()
    xx = x.view(H, 1).repeat(1, W)
    yy = y.repeat(H, 1)
    grid = torch.stack([xx, yy], dim=0)
    return grid
