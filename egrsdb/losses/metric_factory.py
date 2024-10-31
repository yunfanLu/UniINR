import torch
from absl.logging import info
from torch import nn

from egrsdb.losses.global_shutter_reconstruct import GlobalShutterReconstructedMetric
from egrsdb.losses.lpips import LPIPS
from egrsdb.losses.psnr import _PSNR
from egrsdb.losses.ssim import SSIM


def get_single_metric(config):
    if config.NAME == "gs-sharp-psnr":
        return GlobalShutterReconstructedMetric(_PSNR(), input_is_linear=config.input_is_linear, to_rgb=True)
    elif config.NAME == "gs-sharp-ssim":
        return GlobalShutterReconstructedMetric(SSIM(), input_is_linear=config.input_is_linear, to_rgb=True)
    elif config.NAME == "gs-sharp-lpips":
        return GlobalShutterReconstructedMetric(LPIPS(), input_is_linear=config.input_is_linear, to_rgb=True)
    elif config.NAME == "gs-sharp-psnr-linear-domain":
        return GlobalShutterReconstructedMetric(_PSNR())
    elif config.NAME == "gs-sharp-ssim-linear-domain":
        return GlobalShutterReconstructedMetric(SSIM())
    elif config.NAME == "empty":
        return lambda x: 0.0
    # other
    elif config.NAME == "empty":
        return EmptyMetric(config)
    else:
        raise ValueError(f"Unknown metric: {config.NAME}")


class EmptyMetric(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        info(f"EmptyMetric:")
        info(f"  config: {config}")

    def forward(self, batch):
        return torch.tensor(0.0, requires_grad=True)


class MixedMetric(nn.Module):
    def __init__(self, configs):
        super(MixedMetric, self).__init__()
        self.metric = []
        self.eval = []
        for config in configs:
            self.metric.append(config.NAME)
            self.eval.append(get_single_metric(config))
        info(f"Init Mixed Metric: {configs}")

    def forward(self, batch):
        r = []
        for m, e in zip(self.metric, self.eval):
            r.append((m, e(batch)))
        return r
