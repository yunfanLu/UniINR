#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/5 11:04
import torch
import torch.nn as nn

from .dcnv3_nchw import DCNv3NCHW


class UNetDVNv3(nn.Module):
    def __init__(
        self, in_channels, out_channels, dcn_groups, offset_scale, dw_kernel_size, center_feature_scale, bias=True
    ):
        super(UNetDVNv3, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.offset_scale = offset_scale
        self.dcn_groups = dcn_groups
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.bias = bias

        self.relu = nn.LeakyReLU(0.1, True)

        self.conv1 = nn.Conv2d(in_channels, 64, 7, 1, 3, bias=bias)
        self.conv2 = nn.Conv2d(64, 64, 7, 1, 3, bias=bias)
        self.dcn2 = self._get_dcnv3_block(64)
        # Down 1
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=2, padding=3)
        self.conv3 = nn.Conv2d(64, 64, 5, 1, 2, bias=bias)
        self.conv4 = nn.Conv2d(64, 64, 5, 1, 2, bias=bias)
        self.dcn4 = self._get_dcnv3_block(64)
        # Down 2
        self.avgpool2 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1, bias=bias)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        self.dcn6 = self._get_dcnv3_block(128)
        # Down 3
        self.avgpool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, 1, 1, bias=bias)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.dcn8 = self._get_dcnv3_block(256)
        # Down 4
        self.avgpool4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1, bias=bias)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.dcn10 = self._get_dcnv3_block(512)
        # Down 5
        self.avgpool5 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.dcn12 = self._get_dcnv3_block(512)
        # Decoder
        self.upsample2D = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.dcn14 = self._get_dcnv3_block(512)

        self.conv15 = nn.Conv2d(512, 256, 3, 1, 1, bias=bias)
        self.conv16 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.dcn16 = self._get_dcnv3_block(256)

        self.conv17 = nn.Conv2d(256, 128, 3, 1, 1, bias=bias)
        self.conv18 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        self.dcn18 = self._get_dcnv3_block(128)

        self.conv19 = nn.Conv2d(128, 64, 3, 1, 1, bias=bias)
        self.conv20 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.dcn20 = self._get_dcnv3_block(64)

        self.conv21 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.conv22 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.dcn22 = self._get_dcnv3_block(64)

        self.conv23 = nn.Conv2d(64, out_channels, 3, 1, 1, bias=bias)
        self.dcn23 = self._get_dcnv3_block(out_channels)

    def _get_dcnv3_block(self, channels):
        return DCNv3NCHW(
            channels=channels,
            groups=self.dcn_groups,
            offset_scale=self.offset_scale,
            act_layer="GELU",
            norm_layer="LN",
            dw_kernel_size=self.dw_kernel_size,
            center_feature_scale=self.center_feature_scale,
        )

    def forward(self, events, blur_frame):
        X = torch.cat([events, blur_frame], dim=1)

        sources = []
        ## Encoder
        X = self.conv1(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.dcn2(X)
        sources.append(X)

        X = self.avgpool1(X)
        X = self.conv3(X)
        X = self.relu(X)
        X = self.conv4(X)
        X = self.relu(X)
        X = self.dcn4(X)
        sources.append(X)

        X = self.avgpool2(X)
        X = self.conv5(X)
        X = self.relu(X)
        X = self.conv6(X)
        X = self.relu(X)
        X = self.dcn6(X)
        sources.append(X)

        X = self.avgpool3(X)
        X = self.conv7(X)
        X = self.relu(X)
        X = self.conv8(X)
        X = self.relu(X)
        X = self.dcn8(X)
        sources.append(X)

        X = self.avgpool4(X)
        X = self.conv9(X)
        X = self.relu(X)
        X = self.conv10(X)
        X = self.relu(X)
        X = self.dcn10(X)
        sources.append(X)

        X = self.avgpool5(X)
        X = self.conv11(X)
        X = self.relu(X)
        X = self.conv12(X)
        X = self.relu(X)
        X = self.dcn12(X)

        ## Decoder
        X = self.upsample2D(X)
        X = self.conv13(X)
        X = self.relu(X)
        X = X + sources[-1]
        X = self.conv14(X)
        X = self.relu(X)
        X = self.dcn14(X)

        X = self.upsample2D(X)
        X = self.conv15(X)
        X = self.relu(X)
        X = X + sources[-2]
        X = self.conv16(X)
        X = self.relu(X)
        X = self.dcn16(X)

        X = self.upsample2D(X)
        X = self.conv17(X)
        X = self.relu(X)
        X = X + sources[-3]
        X = self.conv18(X)
        X = self.relu(X)
        X = self.dcn18(X)

        X = self.upsample2D(X)
        X = self.conv19(X)
        X = self.relu(X)
        X = X + sources[-4]
        X = self.conv20(X)
        X = self.relu(X)
        X = self.conv20(X)

        X = self.upsample2D(X)
        X = self.conv21(X)
        X = self.relu(X)
        X = X + sources[-5]
        X = self.conv22(X)
        X = self.relu(X)
        X = self.conv22(X)
        X = self.dcn22(X)

        X = self.conv23(X)
        X = self.dcn23(X)
        out = X
        return out
