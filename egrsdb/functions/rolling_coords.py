#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/1/30 14:18

import torch


def get_wht_global_shutter_coordinate(t: float, h: int, w: int):
    """
    :param t: The timestamp of the sampled frame.
    :param h: The height of the sampled frame, which is not the original height.
    :param w: The width of the sampled frame, which is not the original width.
    :return: a cords tensor of shape (1, h, w, 3), where the last dimension
             is (t, w, h).
    """
    assert 0 <= t <= 1, f"Time t should be in [-1, 1], but got {t}."
    # (1, h, w, 3):
    #   1 means one time stamps.
    #   h and w means the height and width of the image.
    #   3 means the w, h, and t coordinates. The order is important.
    grid_map = torch.zeros(1, h, w, 3) + t
    h_coords = torch.linspace(-1, 1, h)
    w_coords = torch.linspace(-1, 1, w)
    mesh_h, mesh_w = torch.meshgrid([h_coords, w_coords])
    # The feature is H W T, so the coords order is (t, w, h)
    # grid_map \in R^{1, h, w, 3}, grid_map[:, :, :, i] is (t, w, h)
    # grid_map[:, :, :, 1:] = torch.stack((mesh_w, mesh_h), 2)
    grid_map[:, :, :, 1:] = torch.stack((mesh_h, mesh_w), 2)
    return grid_map.float()


def get_wht_rolling_shutter_coordinate(t_start: float, t_end: float, h: int, w: int):
    """
    :param t_start: the start time of the rolling shutter.
    :param t_end: the end time of the rolling shutter.
    :param t: The timestamp of the sampled frame.
    :param h: The height of the sampled frame, which is not the original height.
    :param w: The width of the sampled frame, which is not the original width.
    :return: a coordinate tensor of shape (1, h, w, 3), where the last dimension
             is (t, w, h).
    """
    assert 0 <= t_start <= t_end <= 1
    # (1, h, w, 3):
    #   1 means one time stamps.
    #   h and w means the height and width of the image.
    #   3 means the w, h, and t coordinates. The order is important.
    grid_map = torch.zeros(1, h, w, 3) + t_start

    for i in range(h):
        grid_map[0, i, :, 0] = t_start + (t_end - t_start) * i / (h - 1)

    h_coords = torch.linspace(-1, 1, h)
    w_coords = torch.linspace(-1, 1, w)
    mesh_h, mesh_w = torch.meshgrid([h_coords, w_coords])
    # The feature is H W T, so the coords order is (t, w, h)
    # grid_map \in R^{1, h, w, 3}, grid_map[:, :, :, i] is (t, w, h)

    grid_map[:, :, :, 1:] = torch.stack((mesh_h, mesh_w), 2)
    return grid_map.float()


def get_t_global_shutter_coordinate(t: float, h: int, w: int, with_position: bool):
    """
    get a coordinate for an global shutter image.
    :param t: the timestamp of the global shutter image.
    :param h: the height of image
    :param w: the width of image
    :return:
    """
    assert 0 <= t <= 1
    grid_map = torch.zeros(h, w) + t
    if with_position:
        h_coords = torch.linspace(0, 1, h)
        w_coords = torch.linspace(0, 1, w)
        mesh_h, mesh_w = torch.meshgrid([h_coords, w_coords])
        grid_map = torch.stack((mesh_h, mesh_w, grid_map), 2)
        # H, W, C -> C, H, W
        grid_map = grid_map.permute(2, 0, 1)
    else:
        grid_map = grid_map.unsqueeze(0)
    return grid_map.float()


def get_t_rolling_shutter_coordinate(t_start: float, t_end: float, h: int, w: int, with_position: bool):
    """
    get a coordinate for an rolling shutter image.
    :param t_start: the start time of the rolling shutter.
    :param t_end: the end time of the rolling shutter.
    :param h: the height of image
    :param w: the width of image
    :return:
    """
    assert 0 <= t_start <= t_end <= 1
    grid_map = torch.zeros(h, w) + t_start
    for i in range(h):
        grid_map[i, :] = t_start + (t_end - t_start) * i / (h - 1)
    if with_position:
        h_coords = torch.linspace(0, 1, h)
        w_coords = torch.linspace(0, 1, w)
        mesh_h, mesh_w = torch.meshgrid([h_coords, w_coords])
        grid_map = torch.stack((mesh_h, mesh_w, grid_map), 2)
        grid_map = grid_map.permute(2, 0, 1)
    else:
        grid_map = grid_map.unsqueeze(0)
    return grid_map.float()
