#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project :EG-VFI
@Author  :Yunfan Lu (yunfanlu@ust.hk)
@Date    :9/12/2022 15:02
"""

from importlib import import_module

import torch.optim.lr_scheduler as lr_scheduler
from absl.logging import info
from torch.optim import SGD, Adam


class Optimizer:
    def __init__(self, config, model):
        # 1. create optimizer
        if config.NAME == "Adam":
            self.optimizer = Adam(model.parameters(), lr=config.LR)
        elif config.NAME == "SGD":
            self.optimizer = SGD(model.parameters(), lr=config.LR)
        elif config.NAME == "Adam-LIIF":
            liif_decoder_params = model.decoder.spatial_embedding.parameters()
            other_params = [
                param for name, param in model.named_parameters() if "decoder.spatial_embedding" not in name
            ]
            self.optimizer = Adam(
                [
                    {"params": liif_decoder_params, "lr": config.LIIF_DECODER_LR},
                    {"params": other_params, "lr": config.LR},
                ]
            )
        elif config.NAME == "Adam-LIIF-Freeze":
            # liif_decoder_params = model.decoder.spatial_embedding.parameters()
            # other_params = [param for name, param in model.named_parameters() if "decoder.spatial_embedding" not in name]
            for name, param in model.named_parameters():
                if "decoder.spatial_embedding" not in name:
                    param.requires_grad = False
                    info(f"Freeze {name}")
                else:
                    info(f"Unfreeze {name}")
            self.optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LIIF_DECODER_LR)
        else:
            raise ValueError(f"Unknown Optimizer config: {config}")

        # 2. create scheduler
        if config.LR_SCHEDULER == "multi_step":
            self.scheduler = lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=config.milestones,
                gamma=config.decay_gamma,
            )
        elif config.LR_SCHEDULER == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.end_epoch, eta_min=1e-8)
        elif config.LR_SCHEDULER == "cosine_w":
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-8)
        else:
            raise ValueError(f"Unknown Optimizer config: {config}")

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.scheduler.step()

    @property
    def param_groups(self):
        return self.optimizer.param_groups
