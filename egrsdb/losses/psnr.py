import torch
from absl.logging import info
from torch import nn


def almost_equal(x, y):
    return abs(x - y) < 1e-6


class _PSNR(nn.Module):
    def __init__(self):
        super(_PSNR, self).__init__()
        self.eps = torch.tensor(1e-10)

        info(f"Init PSNR:")
        info(f"  Note: the psnr max value is {-10 * torch.log10(self.eps)}")

    def forward(self, x, y):
        d = x - y
        mse = torch.mean(d * d) + self.eps
        psnr = -10 * torch.log10(mse)
        return psnr
