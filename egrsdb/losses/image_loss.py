import torch
from torch.nn.modules.loss import _Loss

from egrsdb.losses.lpips import LPIPS


class L1CharbonnierLossColor(_Loss):
    def __init__(self):
        super(L1CharbonnierLossColor, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        diff_sq = diff * diff
        error = torch.sqrt(diff_sq + self.eps)
        loss = torch.mean(error)
        return loss


class RollingShutterBlurReconstructedLoss(_Loss):
    def __init__(self):
        super(RollingShutterBlurReconstructedLoss, self).__init__()
        self.loss = L1CharbonnierLossColor()

    def forward(self, batch):
        rs_blur_image = batch["rolling_blur_frame_gray"]
        rs_blur_reconstructed_image = batch["rolling_blur_pred_frame"]
        return self.loss(rs_blur_image, rs_blur_reconstructed_image)


class GlobalShutterReconstructedLoss(_Loss):
    def __init__(self, loss_type):
        super(GlobalShutterReconstructedLoss, self).__init__()
        if loss_type == "charbonnier":
            self.loss = L1CharbonnierLossColor()
        elif loss_type == "mse":
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, batch):
        # B, N, C, H, W
        global_sharp_frames = batch["global_sharp_frames"]
        # B, N, C, H, W
        global_sharp_pred_frames = batch["global_sharp_pred_frames"]
        loss = self.loss(global_sharp_frames, global_sharp_pred_frames)
        return loss
