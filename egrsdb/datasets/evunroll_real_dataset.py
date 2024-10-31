import logging
import os
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch
from absl.logging import info
from einops import repeat
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import transforms

from egrsdb.datasets.basic_batch import get_rs_deblur_inference_batch

logging.getLogger("PIL").setLevel(logging.WARNING)


def get_evunroll_real_dataset(fps, data_root, moments, is_color):
    all_seqs = os.listdir(data_root)
    all_seqs.sort()
    seq_dataset_list = []
    for seq in all_seqs:
        if not (seq[:2] in ["01", "02", "03", "04"]):
            continue
        if os.path.isdir(os.path.join(data_root, seq)):
            seq_dataset_list.append(__Sequence_Real(fps, data_root, moments, seq, is_color))
    return ConcatDataset(seq_dataset_list[::-1])


class __Sequence_Real(Dataset):
    def __init__(self, fps, data_root, moments, seq_name, is_color):
        self.seq_name = seq_name
        self.seq = seq_name.split("_")[0]
        self.fps = fps
        self.exposure_time = int(seq_name.split("_")[-2])
        self.delay_time = int(seq_name.split("_")[-1])
        self.whole_time = self.exposure_time + self.delay_time
        self.interval_time = int(1e6 / self.fps)
        self.img_folder = os.path.join(data_root, seq_name, "image")
        self.img_list = sorted(os.listdir(self.img_folder))

        for npz_file in listdir(join(data_root, seq_name)):
            if npz_file.endswith("npz"):
                self.event_file = join(data_root, seq_name, npz_file)

        self.num_input = len(self.img_list)
        im0 = cv2.imread(os.path.join(self.img_folder, self.img_list[0]))
        self.height, self.width, _ = im0.shape
        self.voxel_grid_channel = self.width
        self.moments = moments
        self.ev_idx = None
        self.events = None
        self.is_color = is_color
        self.center_cropped_height = 256
        self.random_cropped_width = 256
        self.info()
        #
        self.to_tensor = transforms.ToTensor()

    def info(self):
        info(f"Init {__class__.__name__}")
        info(f"  seq_name: {self.seq_name}")
        info(f"  seq: {self.seq}")
        info(f"  fps: {self.fps}")
        info(f"  exposure_time: {self.exposure_time}")
        info(f"  delay_time: {self.delay_time}")
        info(f"  whole_time: {self.whole_time}")
        info(f"  interval_time: {self.interval_time}")
        info(f"  img_folder: {self.img_folder}")
        info(f"  img_list: {len(self.img_list)} from {self.img_list[0]} to {self.img_list[-1]}")
        info(f"  event_file: {self.event_file}")
        info(f"  num_input: {self.num_input}")
        info(f"  height: {self.height}")
        info(f"  width: {self.width}")
        info(f"  voxel_grid_channel: {self.voxel_grid_channel}")
        info(f"  moments: {self.moments}")
        info(f"  is_color: {self.is_color}")

    def events_to_rs_voxel_grid(self, event):
        width, height = self.width, self.height
        delay = float(self.delay_time) / (self.height - 1)
        et = event[:, 0].to(torch.float32)
        ex = event[:, 1].long()
        ey = event[:, 2].long()
        ep = event[:, 3].to(torch.float32)
        ep[ep == 0] = -1
        gs_ch = (et / self.whole_time * self.voxel_grid_channel).long()
        gs_events = torch.zeros((self.voxel_grid_channel, height, width), dtype=torch.float32)
        gs_valid = (gs_ch >= 0) & (gs_ch < self.voxel_grid_channel)
        gs_events.index_put_((gs_ch[gs_valid], ey[gs_valid], ex[gs_valid]), ep[gs_valid], accumulate=False)
        return gs_events

    def __len__(self):
        return self.num_input

    def __getitem__(self, index):
        if self.events is None:
            self.events = np.load(self.event_file)["event"]
        # Image
        image_path = os.path.join(self.img_folder, self.img_list[index])
        if self.is_color:
            img_input = Image.open(image_path).convert("RGB")
        else:
            img_input = Image.open(image_path).convert("L")
        img_input = self.to_tensor(img_input)
        # Events
        event_stream = self.get_event(index)
        event_stream = torch.from_numpy(event_stream)
        event_voxel_grid = self.events_to_rs_voxel_grid(event_stream)
        event_voxel_grid[event_voxel_grid < 0] = -1
        event_voxel_grid[event_voxel_grid > 0] = 1
        event_moment = self._to_moments(event_voxel_grid)

        img_input, event_moment = self._crop(img_input, event_moment)
        # Batch
        sample = get_rs_deblur_inference_batch()
        sample["video_name"] = self.seq_name
        sample["rolling_blur_frame_name"] = image_path.split("/")[-1].split(".")[0]
        if self.is_color:
            sample["rolling_blur_frame_color"] = img_input  # type: ignore
        else:
            sample["rolling_blur_frame_gray"] = img_input  # type: ignore
        sample["events"] = event_moment
        return sample

    def _to_moments(self, events):
        B, H, W = events.shape
        windows_size = B // self.moments
        if B % self.moments > 0:
            windows_size += 1
        events_zeros = torch.zeros((self.moments * windows_size, H, W))
        events_zeros[:B, :, :] = events
        events_zeros = events_zeros.view(self.moments, windows_size, H, W)
        events_zeros = events_zeros.mean(dim=1)
        return events_zeros

    def _crop(self, input_frame, events):
        # Drop the height
        drop_size_height = self.height - self.center_cropped_height
        th = drop_size_height // 2
        min_x = th
        max_x = min_x + self.center_cropped_height
        # Drop the width
        min_y = (self.width - self.random_cropped_width) // 2
        max_y = min_y + self.random_cropped_width
        # Crop
        input_frames = input_frame[:, min_x:max_x, min_y:max_y]
        events = events[:, min_x:max_x, min_y:max_y]
        return input_frames, events

    def get_timemap(self, sample):
        row_stamp = torch.arange(self.height, dtype=torch.float32) / (
            self.height - 1
        ) * self.delay_time / self.whole_time + self.exposure_time / (2 * self.whole_time)
        target_dis = row_stamp - sample["timestamp"]

        time_map = torch.stack([row_stamp, target_dis], dim=1)
        time_map = repeat(time_map, "h c-> h w c", w=self.width)

        sample["time_map"] = time_map
        return sample

    def get_event(self, idx):
        if self.ev_idx is None:
            if self.events.ndim == 1:
                et = self.events["t"]
                ex = self.events["x"]
                ey = self.events["y"]
                ep = self.events["p"]
                self.events = np.stack([et, ex, ey, ep], axis=1)
            self.ev_idx = []
            ev_start_idx = 0
            ev_end_idx = 0
            for i in range(self.num_input):
                start_t = self.interval_time * i
                end_t = start_t + self.whole_time

                ev_start_idx = ev_end_idx
                while self.events[ev_start_idx, 0] < start_t:
                    ev_start_idx += 1
                ev_end_idx = ev_start_idx
                while self.events[ev_end_idx, 0] < end_t:
                    ev_end_idx += 1
                self.ev_idx.append((ev_start_idx, ev_end_idx))

        start_idx, end_idx = self.ev_idx[idx]
        event = self.events[start_idx:end_idx].copy()
        event[:, 0] = event[:, 0] - self.interval_time * idx
        return event  # (115718, 4)