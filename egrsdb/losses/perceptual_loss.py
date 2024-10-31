#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/13 17:15

import torch.nn as nn
import torchvision.models as models
from torch.nn.modules.loss import _Loss


class PerceptualLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.content_func = self._vgg_3_3()
        self.criterion = nn.L1Loss()

    @staticmethod
    def _vgg_3_3():
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def forward(self, fake, real):
        f_fake = self.content_func.forward(fake)
        f_real = self.content_func.forward(real)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
