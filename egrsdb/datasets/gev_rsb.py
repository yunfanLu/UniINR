#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/12/24 16:23
import logging
import random
from os import listdir
from os.path import isdir, join

import numpy as np
import torch
from absl.logging import info
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from egrsdb.datasets.basic_batch import get_rs_deblur_inference_batch
from egrsdb.functions.image_domain_converting import rgb_to_lineal
from egrsdb.functions.timestamp import get_gs_sharp_timestamps

logging.getLogger("PIL").setLevel(logging.WARNING)

TEST_VIDEO_NAMES = [
    "24209_1_13",
    "24209_1_28",
    "24209_1_31",
    "24209_1_36",
    "24209_1_38",
    "24209_1_25",
    "24209_1_30",
    "24209_1_33",
    "24209_1_37",
]

TRAIN_VIDEO_NAMES = [
    "24209_1_10",
    "24209_1_21",
    "24209_1_26",
    "24209_1_39",
    "24209_1_42",
    "24209_1_11",
    "24209_1_22",
    "24209_1_29",
    "24209_1_4",
    "24209_1_5",
    "24209_1_12",
    "24209_1_23",
    "24209_1_32",
    "24209_1_40",
    "24209_1_6",
    "24209_1_14",
    "24209_1_24",
    "24209_1_35",
    "24209_1_41",
    "24209_1_9",
]


def get_gev_rolling_shutter_blur_dataset(
    root,
    blur_accumulate,
    events_moment,
    gs_sharp_frame_count,
    center_cropped_height,
    random_cropped_width,
    is_color,
    gs_sharp_start_index,
    gs_sharp_end_index,
    calculate_in_linear_domain,
    event_for_gs_frame_buffer,
    correct_offset,
):
    image_height = 260
    image_width = 346
    # Inputs:
    gs_frame_total = image_height + blur_accumulate
    # the start time, end time and exposure of rolling shutter blur.
    input_rs_blur_times = [0, image_height / gs_frame_total, blur_accumulate / gs_frame_total]
    # Outputs:
    gs_sharp_indexes, gs_sharp_timestamps = get_gs_sharp_timestamps(
        gs_sharp_start_index,
        gs_sharp_end_index,
        gs_sharp_frame_count,
        total=gs_frame_total,
        correct_offset=correct_offset,
    )
    info(f"get_gev_rolling_shutter_blur_dataset:")
    info(f"  gs_sharp_frame_count: {gs_sharp_frame_count}")
    info(f"  gs_sharp_start_index: {gs_sharp_start_index}")
    info(f"  gs_sharp_end_index: {gs_sharp_end_index}")
    info(f"  gs_frame_total: {gs_frame_total}")
    info(f"  correct_offset: {correct_offset}")
    info(f"  gs_sharp_indexes: {gs_sharp_indexes}")
    info(f"  gs_sharp_timestamps: {gs_sharp_timestamps}")

    return _get_gev_rs_dataset(
        root=root,
        blur_accumulate=blur_accumulate,
        input_rs_blur_frame_timestamp=input_rs_blur_times,  # type: ignore
        output_gs_sharp_frame_indexes=gs_sharp_indexes,
        output_gs_sharp_frame_timestamps=gs_sharp_timestamps,
        events_moment=events_moment,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
        image_height=image_height,
        image_width=image_width,
        is_color=is_color,
        calculate_in_linear_domain=calculate_in_linear_domain,
        event_for_gs_frame_buffer=event_for_gs_frame_buffer,
    )


def _get_gev_rs_dataset(
    root,
    blur_accumulate,
    input_rs_blur_frame_timestamp,
    output_gs_sharp_frame_indexes,
    output_gs_sharp_frame_timestamps,
    events_moment,
    center_cropped_height,
    random_cropped_width,
    image_height,
    image_width,
    is_color,
    calculate_in_linear_domain,
    event_for_gs_frame_buffer,
):
    train = GevRollingShutterBlurDataset(
        root=root,
        blur_accumulate=blur_accumulate,
        input_rs_blur_frame_timestamp=input_rs_blur_frame_timestamp,
        output_gs_sharp_frame_indexes=output_gs_sharp_frame_indexes,
        output_gs_sharp_frame_timestamps=output_gs_sharp_frame_timestamps,
        events_moment=events_moment,
        is_train=True,
        height=image_height,
        width=image_width,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
        is_color=is_color,
        calculate_in_linear_domain=calculate_in_linear_domain,
        event_for_gs_frame_buffer=event_for_gs_frame_buffer,
    )
    test = GevRollingShutterBlurDataset(
        root=root,
        blur_accumulate=blur_accumulate,
        input_rs_blur_frame_timestamp=input_rs_blur_frame_timestamp,
        output_gs_sharp_frame_indexes=output_gs_sharp_frame_indexes,
        output_gs_sharp_frame_timestamps=output_gs_sharp_frame_timestamps,
        events_moment=events_moment,
        is_train=False,
        height=image_height,
        width=image_width,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
        is_color=is_color,
        calculate_in_linear_domain=calculate_in_linear_domain,
        event_for_gs_frame_buffer=event_for_gs_frame_buffer,
    )
    return train, test


class GevRollingShutterBlurDataset(Dataset):
    def __init__(
        self,
        root,
        blur_accumulate,
        input_rs_blur_frame_timestamp,
        output_gs_sharp_frame_indexes,
        output_gs_sharp_frame_timestamps,
        events_moment,
        is_train,
        height,
        width,
        center_cropped_height,
        random_cropped_width,
        is_color,
        calculate_in_linear_domain,
        event_for_gs_frame_buffer,
    ):
        super(GevRollingShutterBlurDataset, self).__init__()
        # check the input parameters.
        assert center_cropped_height % 32 == 0, f"center_cropped_height {center_cropped_height} % 32 != 0"
        assert random_cropped_width % 32 == 0, f"random_cropped_width {random_cropped_width} % 32 != 0"
        assert center_cropped_height <= height, f"center_cropped_height {center_cropped_height} > height {height}"
        assert random_cropped_width <= width, f"random_cropped_width {random_cropped_width} > width {width}"
        assert 520 % events_moment == 0, f"events_moment_count 520 % {events_moment} != 0"
        # Set the global configuration.
        self.root = root
        # Input: Rolling shutter blur frames and events.
        self.blur_accumulate = blur_accumulate
        self.input_rs_blur_frame_timestamp = input_rs_blur_frame_timestamp
        self.output_gs_sharp_frame_indexes = output_gs_sharp_frame_indexes
        self.output_gs_sharp_frame_timestamps = output_gs_sharp_frame_timestamps
        self.events_moment_count = events_moment
        self.event_for_gs_frame_buffer = event_for_gs_frame_buffer
        self.is_color = is_color
        self.calculate_in_linear_domain = calculate_in_linear_domain
        # The training config.
        self.is_train = is_train
        self.H = height
        self.W = width
        self.center_cropped_height = center_cropped_height
        self.random_cropped_width = random_cropped_width
        # walk for all items
        self.samples = self._walk()
        # Transformer
        self.to_tensor = transforms.ToTensor()
        self._info()

    def _info(self):
        info(f"Dataset: {self.__class__.__name__}")
        info(f"  Root: {self.root}")
        info(f"  blur_accumulate: {self.blur_accumulate}")
        info(f"  input_rs_blur_frame_timestamp: {self.input_rs_blur_frame_timestamp}")
        info(f"  output_gs_sharp_frame_indexes:  {self.output_gs_sharp_frame_indexes}")
        info(f"  output_gs_sharp_frame_timestamp: {self.output_gs_sharp_frame_timestamps}")
        info(f"  events_moment_count: {self.events_moment_count}")
        info(f"  Is train: {self.is_train}")
        info(f"  Height: {self.H}")
        info(f"  Width: {self.W}")
        info(f"  Samples: {len(self.samples)}")
        info(f"  Crop height: {self.center_cropped_height}")
        info(f"  Crop width: {self.random_cropped_width}")
        info(f"  self.calculate_in_linear_domain: {self.calculate_in_linear_domain}")

    def __getitem__(self, index):
        item = self.samples[index]
        rolling_blur_frame, input_events, all_global_sharp_frames = item
        # get output global sharp frames
        output_global_sharp_frames = []
        for idx in self.output_gs_sharp_frame_indexes:
            output_global_sharp_frames.append(all_global_sharp_frames[idx])
        # Load data
        video_name = rolling_blur_frame.split("/")[-2]
        rolling_frame_name = rolling_blur_frame.split("/")[-1].split(".")[0]
        if self.is_color:
            rolling_frame = Image.open(rolling_blur_frame).convert("RGB")
        else:
            rolling_frame = Image.open(rolling_blur_frame).convert("L")
        rolling_blur_frame = self.to_tensor(rolling_frame)
        # events
        events = [self._load_events(e) for e in input_events]  # [260, 346] x N
        # events_for_gs_frames = [events[i] for i in self.output_gs_sharp_frame_indexes]
        events_for_gs_sharp_frames = []
        for i in self.output_gs_sharp_frame_indexes:
            event_buffer = np.zeros_like(events[0])
            l = max(0, i - self.event_for_gs_frame_buffer)
            r = min(len(events), i + self.event_for_gs_frame_buffer + 1)
            for j in range(l, r):
                event_buffer = event_buffer + events[j]
            event_buffer = event_buffer / (r - l)
            events_for_gs_sharp_frames.append(event_buffer)
        events_for_gs_sharp_frames = np.stack(events_for_gs_sharp_frames, axis=0)  # B1 x 260 x 346
        events_for_gs_sharp_frames = (
            torch.from_numpy(events_for_gs_sharp_frames).unsqueeze(1).float()
        )  # B1 x 1 x 260 x 346
        # global event
        events = np.stack(events, axis=0)  # B x 260 x 346
        events = torch.from_numpy(events).float()  # B x 260 x 346
        events = self._to_moments(events)
        # global sharp frames
        if self.is_color:
            output_global_frames = [self.to_tensor(Image.open(i).convert("RGB")) for i in output_global_sharp_frames]
        else:
            output_global_frames = [self.to_tensor(Image.open(i).convert("L")) for i in output_global_sharp_frames]
        output_global_frames = torch.stack(output_global_frames)
        rs_b_frame, gs_s_frames, events, events_for_gs_sharp_frames = self._crop(
            rolling_blur_frame, output_global_frames, events, events_for_gs_sharp_frames
        )
        # generate the batch
        # 1. item id
        batch = get_rs_deblur_inference_batch()
        batch["video_name"] = video_name
        batch["rolling_blur_frame_name"] = rolling_frame_name
        # 2. input: rolling blur
        if self.calculate_in_linear_domain:
            rs_b_frame = rgb_to_lineal(rs_b_frame)
            gs_s_frames = rgb_to_lineal(gs_s_frames)

        if self.is_color:
            batch["rolling_blur_frame_color"] = rs_b_frame  # type: ignore
        else:
            batch["rolling_blur_frame_gray"] = rs_b_frame  # type: ignore
        # 3. input: events
        batch["events"] = events  # type: ignore
        batch["events_for_gs_sharp_frames"] = events_for_gs_sharp_frames  # type: ignore
        batch["global_sharp_frames"] = gs_s_frames  # type: ignore
        batch["global_sharp_frame_timestamps"] = self.output_gs_sharp_frame_timestamps  # type: ignore
        return batch

    def __len__(self):
        return len(self.samples)

    def _crop(self, input_frame, output_global_frames, events, events_for_gs_sharp_frames):
        # Drop the height
        drop_size_height = self.H - self.center_cropped_height
        th = drop_size_height // 2
        min_x = th
        max_x = min_x + self.center_cropped_height
        # Drop the width
        if self.is_train:
            min_y = random.randint(0, self.W - self.random_cropped_width)
        else:
            min_y = (self.W - self.random_cropped_width) // 2
        max_y = min_y + self.random_cropped_width
        # Crop
        input_frames = input_frame[:, min_x:max_x, min_y:max_y]
        output_global_frames = output_global_frames[:, :, min_x:max_x, min_y:max_y]
        events = events[:, min_x:max_x, min_y:max_y]
        events_for_gs_frames = events_for_gs_sharp_frames[:, :, min_x:max_x, min_y:max_y]
        return input_frames, output_global_frames, events, events_for_gs_frames

    def _to_moments(self, events):
        B, H, W = events.shape
        windows_size = B // self.events_moment_count
        if B % self.events_moment_count > 0:
            windows_size += 1
        events_zeros = torch.zeros((self.events_moment_count * windows_size, H, W))
        events_zeros[:B, :, :] = events
        events_zeros = events_zeros.view(self.events_moment_count, windows_size, H, W)
        events_zeros = events_zeros.mean(dim=1)
        return events_zeros

    def _load_events(self, event_path):
        events = np.load(event_path)
        events = self._render(shape=[260, 346], **events)
        return events

    @staticmethod
    def _render(x, y, t, p, shape):
        events = np.zeros(shape=shape)
        events[y, x] = p
        return events

    def _walk(self):
        folders = sorted(listdir(self.root))
        gs_sharp_folders = []
        event_folders = []
        rolling_blur_folders = []
        rolling_sharp_folders = []
        for folder in folders:
            if not isdir(join(self.root, folder)):
                continue
            if folder.endswith("fps5000-resize-frames-H260-W346"):  # global sharp
                gs_sharp_folders.append(folder)
            elif folder.endswith("fps5000-resize-frames-H260-W346-events"):
                event_folders.append(folder)
            elif folder.endswith("fps5000-resize-frames-H260-W346-rolling"):
                rolling_sharp_folders.append(folder)
            elif folder.endswith(f"fps5000-blur-frames-H260-W346-blur-accumulate-{self.blur_accumulate}"):
                rolling_blur_folders.append(folder)

        info(f"  gs_sharp_folders: {len(gs_sharp_folders)}")
        info(f"  event_folders: {len(event_folders)}")
        info(f"  rolling_sharp_folders: {len(rolling_sharp_folders)}")
        info(f"  rolling_blur_folders: {len(rolling_blur_folders)}")
        assert len(gs_sharp_folders) == len(event_folders) == len(rolling_blur_folders) == len(rolling_sharp_folders)

        assert len(gs_sharp_folders) == (len(TEST_VIDEO_NAMES) + len(TRAIN_VIDEO_NAMES))

        items = []
        for gs_sharp_folder, event_folder, rolling_blur_folder, rolling_sharp_folder in zip(
            gs_sharp_folders, event_folders, rolling_blur_folders, rolling_sharp_folders
        ):
            assert gs_sharp_folder[:10] == event_folder[:10] == rolling_blur_folder[:10] == rolling_sharp_folder[:10]
            video_name = gs_sharp_folder[:10]
            if self.is_train and (video_name in TRAIN_VIDEO_NAMES):
                items.extend(self._walk_video(gs_sharp_folder, event_folder, rolling_blur_folder, rolling_sharp_folder))
            if (not self.is_train) and (video_name in TEST_VIDEO_NAMES):
                items.extend(self._walk_video(gs_sharp_folder, event_folder, rolling_blur_folder, rolling_sharp_folder))
        return items

    def _walk_video(self, gs_sharp_folder, event_folder, rolling_blur_folder, sharp_rolling_folder):
        gs_sharp_frames = sorted(listdir(join(self.root, gs_sharp_folder)))
        event_npz_png = sorted(listdir(join(self.root, event_folder)))
        events = [npz for npz in event_npz_png if npz.endswith(".npz")]
        rolling_blur_frames = sorted(listdir(join(self.root, rolling_blur_folder)))
        rolling_sharp_frames = sorted(listdir(join(self.root, sharp_rolling_folder)))

        # every two adjacent images will generate some events.
        assert len(gs_sharp_frames) - 1 == len(events)
        # The first frame of rolling data is 260.png, which means that a total
        # of 260 images from images 0-256 in the global are synthesized.
        assert len(gs_sharp_frames) == len(rolling_sharp_frames) + self.H
        assert len(rolling_blur_frames) == len(rolling_sharp_frames) // self.H

        items = []
        for i in range(len(rolling_blur_frames)):
            # e.g., 0000000260-0000000520.png
            rolling_blur_name = rolling_blur_frames[i].split(".")[0]
            begin_rs_index = rolling_blur_name.split("-")[0]
            end_rs_index = rolling_blur_name.split("-")[1]
            # Input: Rolling blur
            rolling_blur_path = join(self.root, rolling_blur_folder, rolling_blur_frames[i])
            # Input: Event
            event_start_index = int(begin_rs_index) - self.H
            event_end_index = int(end_rs_index)
            items_events = []
            for j in range(event_start_index, event_end_index):
                items_events.append(join(self.root, event_folder, events[j]))
            # Output: Rolling sharp
            gs_sharp_start_index = int(begin_rs_index) - self.H
            gs_sharp_end_index = int(end_rs_index)
            items_sharp = []
            for j in range(gs_sharp_start_index, gs_sharp_end_index):
                items_sharp.append(join(self.root, gs_sharp_folder, gs_sharp_frames[j]))
            # generate an item
            items.append((rolling_blur_path, items_events, items_sharp))
        return items
