from torch.nn.modules.loss import _Loss

from egrsdb.functions.image_domain_converting import lineal_to_rgb


class GlobalShutterReconstructedMetric(_Loss):
    def __init__(self, metric, input_is_linear=False, to_rgb=False):
        super(GlobalShutterReconstructedMetric, self).__init__()
        self.metric = metric
        self.input_is_linear = input_is_linear
        self.to_rgb = to_rgb

    def forward(self, batch):
        # B, N, C, H, W
        global_sharp_frames = batch["global_sharp_frames"]
        # B, N, C, H, W
        global_sharp_pred_frames = batch["global_sharp_pred_frames"]

        if self.input_is_linear and self.to_rgb:
            global_sharp_frames = lineal_to_rgb(global_sharp_frames)
            global_sharp_pred_frames[global_sharp_pred_frames < 0] = 0
            global_sharp_pred_frames = lineal_to_rgb(global_sharp_pred_frames)

        N = global_sharp_frames.shape[1]
        metric = 0
        for i in range(N):
            gt = global_sharp_frames[:, i, :, :, :]
            pred = global_sharp_pred_frames[:, i, :, :, :]
            metric = metric + self.metric(gt, pred).float()
        return metric / N
