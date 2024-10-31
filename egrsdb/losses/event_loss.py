import math

from absl.logging import info
from torch import nn
from torch.nn.modules.loss import _Loss


class GlobalShutterDifferentialReconstructedLoss(_Loss):
    def __init__(self, loss_type):
        super(GlobalShutterDifferentialReconstructedLoss, self).__init__()
        info(f"GlobalShutterDifferentialReconstructedLoss: loss_type: {loss_type}")
        if loss_type == "l1":
            self.loss = nn.L1Loss()
        elif loss_type == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, batch):
        global_sharp_pred_frames_differential = batch["global_sharp_pred_frames_differential"]
        events_for_gs_sharp_frames = batch["events_for_gs_sharp_frames"]
        loss = self.loss(global_sharp_pred_frames_differential, events_for_gs_sharp_frames)
        return loss
