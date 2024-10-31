from os import listdir
from os.path import isdir, isfile, join

import cv2
import h5py
import numpy as np
import torch
from absl.logging import info
from torch.utils.data import Dataset

from egrsdb.datasets.basic_batch import get_rs_deblur_inference_batch


def get_dre_dataset(config):
    root = config.root
    moments = config.events_moment

    video_folder = listdir(root)
    dataset_list = []
    for video in video_folder:
        video_path = join(root, video)
        if not isdir(video_path):
            continue
        dataset = SingleDavisRollingShutterEventsVideoDataset(video_path, moments)
        dataset_list.append(dataset)
    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset, dataset


class SingleDavisRollingShutterEventsVideoDataset(Dataset):
    def __init__(self, video_root, moments, crop_h=256, crop_w=256):
        super(SingleDavisRollingShutterEventsVideoDataset, self).__init__()
        assert isdir(video_root), f"video_root: {video_root} is not a directory"

        rs_folder = join(video_root, "rs")
        self.rs_images = sorted([join(rs_folder, f) for f in listdir(rs_folder) if isfile(join(rs_folder, f))])

        self.event_file = join(video_root, "event.h5")
        self.has_load_events = False

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.moments = moments

    def _load_events(self):
        with h5py.File(self.event_file) as f:
            self.img_ts = np.asarray(f["image_ts"])
            self.start_img_ts = np.asarray(f["start_image_ts"])
            self.end_img_ts = np.asarray(f["end_image_ts"])
            x = np.asarray(f["x"]).astype(np.int64)
            y = np.asarray(f["y"]).astype(np.int64)
            t = np.asarray(f["t"]).astype(np.int64)
            p = np.asarray(f["p"]).astype(np.int64)
            p[p == 0] = -1

            self.events = np.stack([t, x, y, p], axis=1)

            self.exp_time = self.end_img_ts - self.start_img_ts
            self.delay_time = 70

    def __len__(self):
        return len(self.rs_images)

    def __getitem__(self, idx):
        if not self.has_load_events:
            self._load_events()
            self.has_load_events = True

        rs_image_path = self.rs_images[idx]
        rs_image = cv2.imread(rs_image_path)
        rs_image = cv2.cvtColor(rs_image, cv2.COLOR_BGR2RGB)
        rs_image = torch.from_numpy(rs_image).permute(2, 0, 1).float() / 255.0
        C, H, W = rs_image.shape

        exp = self.exp_time[idx]
        delay = self.delay_time

        rs_start = self.start_img_ts[idx]
        rs_end = self.end_img_ts[idx] + delay * (H - 1) + exp
        events = self._select_events(rs_start, rs_end)
        events = torch.from_numpy(events)
        event_voxel_grid = self.events_to_rs_voxel_grid(events, H, W)

        rs_image, event_voxel_grid = self._crop(rs_image, event_voxel_grid)

        # Batch
        sample = get_rs_deblur_inference_batch()
        sample["video_name"] = rs_image_path.split("/")[-3]
        sample["rolling_blur_frame_name"] = rs_image_path.split("/")[-1].split(".")[0]
        sample["rolling_blur_frame_color"] = rs_image  # type: ignore
        sample["events"] = event_voxel_grid
        return sample

    def _crop(self, input_frame, events):
        ch, cw = self.crop_h, self.crop_w
        input_frames = input_frame[:, 0:ch, 0:cw]
        events = events[:, 0:ch, 0:cw]
        return input_frames, events

    def events_to_rs_voxel_grid(self, event, H, W):
        et = event[:, 0].to(torch.float32)
        ex = event[:, 1].long()
        ey = event[:, 2].long()
        ep = event[:, 3].to(torch.float32)

        time_range = et.max() - et.min()
        gs_ch = ((et - et.min()) / time_range * (self.moments - 1)).long()
        gs_events = torch.zeros((self.moments, H, W), dtype=torch.float32)

        gs_events.index_put_((gs_ch, ey, ex), ep, accumulate=True)
        return gs_events

    def _select_events(self, start, end):
        mask = (self.events[:, 0] >= start) & (self.events[:, 0] < end)
        events = self.events[mask]
        events = events[events[:, 0].argsort()]
        events[:, 0] = events[:, 0] - events[0, 0]
        return events
