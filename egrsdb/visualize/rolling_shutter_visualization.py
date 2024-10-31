import os
from os.path import join

import cv2
import numpy as np
import torch
from absl.logging import debug, flags, info

from egrsdb.utils.flow_viz import flow_to_image

FLAGS = flags.FLAGS


def event_vis(event):
    H, W = event.shape
    event_image = np.zeros((H, W, 3), dtype=np.uint8) + 255
    event_image[event > 0] = [0, 0, 255]
    event_image[event < 0] = [255, 0, 0]
    return event_image


class VisualizationRollingShutter:
    def __init__(self, visualization_config):
        """The visualization class for tesis.

        Args:
            visualization_config (EasyDict): The visualization config of testing.
        """
        self.saving_folder = join(FLAGS.log_dir, visualization_config.folder)
        os.makedirs(self.saving_folder, exist_ok=True)
        self.count = 0
        #
        self.tag = visualization_config.tag
        self.intermediate_visualization = visualization_config.intermediate_visualization
        info("Init Visualization:")
        info(f"  saving_folder: {self.saving_folder}")

    def visualize(self, inputs):
        def _save(image, path):
            if not isinstance(image, torch.Tensor):
                debug(f"Image is not a tensor, but a {type(image)}, now image saved in {path}")
                return
            image = image.detach()
            image = image.permute(1, 2, 0).cpu().numpy()
            image = image.clip(0, 1)
            image = (image * 255).astype(np.uint8)
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image)

        def _save_event(event, path):
            event = event.detach()
            C, H, W = event.shape
            event_sum = event.sum(dim=0).cpu().numpy()
            event_image = event_vis(event_sum)
            cv2.imwrite(path, event_image)
            event = event.cpu().numpy()
            for i in range(min(C, 3)):
                cv2.imwrite(path.replace(".png", f"_{str(i).zfill(3)}.png"), event_vis(event[i]))

        def _save_differential(differential, path):
            if not isinstance(differential, torch.Tensor):
                debug(f"Image is not a tensor, but a {type(differential)}, now image saved in {path}")
                return
            differential = differential.detach()
            differential = differential.permute(1, 2, 0).cpu().numpy()
            # save an differential
            flow = np.zeros((differential.shape[0], differential.shape[1], 2), dtype=np.float32)
            flow = flow + differential
            flow = flow_to_image(flow, convert_to_bgr=True, normalize=True)
            cv2.imwrite(path, flow)

        info(f"video_name: {inputs['video_name']}")
        info(f"rolling_blur_frame_name: {inputs['rolling_blur_frame_name']}")

        B = len(inputs["video_name"])
        for b in range(B):
            video_name = inputs["video_name"][b]
            frame_name = inputs["rolling_blur_frame_name"][b]
            testfolder = join(self.saving_folder, video_name)
            os.makedirs(testfolder, exist_ok=True)
            # save input
            # save input event
            _save_event(inputs["events"][b], join(testfolder, f"{frame_name}_event.png"))
            # save input blur rolling shutter image
            _save(inputs["rolling_blur_frame_gray"][b], join(testfolder, f"{frame_name}_rsb_gray.png"))
            _save(inputs["rolling_blur_frame_color"][b], join(testfolder, f"{frame_name}_rsb_color.png"))
            # output
            _save(inputs["rolling_blur_pred_frame"][b], join(testfolder, f"{frame_name}_rsb_pred.png"))
            # save gt
            if isinstance(inputs["global_sharp_frames"], torch.Tensor):
                B, N = inputs["global_sharp_frames"].shape[:2]
                for i in range(N):
                    gss_gt = inputs["global_sharp_frames"][b, i]
                    _save(gss_gt, join(testfolder, f"{frame_name}_{str(i).zfill(3)}_gss_gt.png"))

            if isinstance(inputs["events_for_gs_sharp_frames"], torch.Tensor):
                B, EN = inputs["events_for_gs_sharp_frames"].shape[:2]
                for i in range(EN):
                    event = inputs["events_for_gs_sharp_frames"][b, i]
                    _save_event(event, join(testfolder, f"{frame_name}_{str(i).zfill(3)}_gss_event.png"))

            # save global sharp image
            # inputs["global_sharp_pred_frames"] = [B, N, C, H, W]
            if isinstance(inputs["global_sharp_pred_frames"], torch.Tensor):
                B, N = inputs["global_sharp_pred_frames"].shape[:2]
                for i in range(N):
                    gss = inputs["global_sharp_pred_frames"][b, i]
                    _save(gss, join(testfolder, f"{frame_name}_{str(i).zfill(3)}_gss_pred.png"))
            # save rolling shutter sharp image
            if isinstance(inputs["rolling_sharp_pred_frames"], torch.Tensor):
                B, N = inputs["rolling_sharp_pred_frames"].shape[:2]
                for i in range(N):
                    rss = inputs["rolling_sharp_pred_frames"][b, i]
                    _save(rss, join(testfolder, f"{frame_name}_{str(i).zfill(3)}_rss_pred.png"))
            # save dofference
            if isinstance(inputs["global_sharp_pred_frames_differential"], torch.Tensor):
                B, N = inputs["global_sharp_pred_frames_differential"].shape[:2]
                for i in range(N):
                    gss_differential = inputs["global_sharp_pred_frames_differential"][b, i]
                    _save_differential(
                        gss_differential, join(testfolder, f"{frame_name}_{str(i).zfill(3)}_gss_differential.png")
                    )
