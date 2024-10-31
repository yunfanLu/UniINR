from torch import nn

try:
    from ops_dcnv3.modules import DCNv3
except:
    pass


class DCNv3NCHW(nn.Module):
    def __init__(self, channels, groups, offset_scale, act_layer, norm_layer, dw_kernel_size, center_feature_scale):
        super(DCNv3NCHW, self).__init__()
        self.dcn = DCNv3(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
            center_feature_scale=center_feature_scale,
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.dcn(x)
        x = x.permute(0, 3, 1, 2)
        return x
