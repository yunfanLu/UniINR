import torch
import torch.nn as nn


class _SCN(nn.Module):
    def __init__(self):
        super(_SCN, self).__init__()
        self.W1 = nn.Conv2d(40, 128, 3, 1, 1, bias=False)
        self.S1 = nn.Conv2d(128, 40, 3, 1, 1, groups=1, bias=False)
        self.S2 = nn.Conv2d(40, 128, 3, 1, 1, groups=1, bias=False)
        self.shlu = nn.ReLU(True)

    def forward(self, input):
        x1 = input[:, range(0, 40), :, :]
        event_input = input[:, range(40, 80), :, :]

        x1 = torch.mul(x1, event_input)
        z = self.W1(x1)
        tmp = z
        for i in range(25):
            ttmp = self.shlu(tmp)
            x = self.S1(ttmp)
            x = torch.mul(x, event_input)
            x = torch.mul(x, event_input)
            x = self.S2(x)
            x = ttmp - x
            tmp = torch.add(x, z)
        c = self.shlu(tmp)
        return c


class ESL(nn.Module):
    def __init__(self, scale, is_color):
        super(ESL, self).__init__()
        assert scale in [1, 2, 4], "scale must be 2 or 4"
        self.scale = scale

        in_channel = 3 if is_color else 1
        self.in_channel = in_channel

        self.scn = nn.Sequential(_SCN())

        self.image_d = nn.Conv2d(
            in_channels=in_channel,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.event_c1 = nn.Conv2d(
            in_channels=40,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.event_c2 = nn.Conv2d(
            in_channels=40,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

        if self.scale >= 2:
            self.shu1 = nn.Conv2d(
                in_channels=128,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.ps1 = nn.PixelShuffle(2)

        if self.scale == 4:
            self.shu2 = nn.Conv2d(
                in_channels=128,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.ps2 = nn.PixelShuffle(2)

        self.end_conv = nn.Conv2d(
            in_channels=128,
            out_channels=in_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, batch):
        # load data
        if self.in_channel == 1:
            image = batch["rolling_blur_frame_gray"]
        else:
            image = batch["rolling_blur_frame_color"]
        event = batch["events"]
        # for inference
        x1 = image
        x1 = self.image_d(x1)

        event_out = self.event_c1(event)
        event_out = torch.sigmoid(event_out)
        event_out = self.event_c2(event_out)
        event_out = torch.sigmoid(event_out)
        scn_input = torch.cat([x1, event_out], 1)
        out = self.scn(scn_input)

        if self.scale >= 2:
            out = self.shu1(out)
            out = self.ps1(out)
        if self.scale == 4:
            out = self.shu2(out)
            out = self.ps2(out)

        out = self.end_conv(out)
        #
        batch["global_sharp_pred_frames"] = out.unsqueeze(1)
        return batch


def get_elsnet(scale, is_color):
    return ESL(scale, is_color)
