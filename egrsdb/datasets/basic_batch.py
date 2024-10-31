#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/12/24 15:57
from enum import Enum, unique


def get_rs_deblur_inference_batch():
    return {
        "video_name": "NONE",
        "rolling_blur_frame_name": "NONE",
        # input: events
        "events": "NONE",
        "events_for_gs_sharp_frames": "NONE",
        # input: rolling blur frame
        # color image only for reference
        "rolling_blur_frame_color": "NONE",
        # input gray image
        "rolling_blur_frame_gray": "NONE",
        "rolling_blur_start_time": "NONE",
        "rolling_blur_end_time": "NONE",
        "rolling_blur_exposure_time": "NONE",
        # output: rolling blur frame
        "rolling_sharp_pred_frames": "NONE",
        "rolling_blur_pred_frame": "NONE",
        # input: global sharp frame
        "global_sharp_frame_timestamps": "NONE",
        "global_sharp_frames": "NONE",
        # Output: global sharp frame
        "global_sharp_pred_frames": "NONE",
        "global_sharp_pred_frames_differential": "NONE",  # List[] N x B
    }
