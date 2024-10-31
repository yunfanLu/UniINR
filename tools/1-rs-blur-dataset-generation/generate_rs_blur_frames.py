# encoding: UTF-8
# author:   yunfan lu (yunfanlu at foxmail dot com)

from os import listdir, makedirs
from os.path import join

import cv2
import numpy as np
from absl import app, flags
from absl.logging import info
from pudb import set_trace

"""
This is the script to generate the RS blur frames for the EvUnrall or Fastec dataset.
The original dataset structure is as follows:
.
|-- 24209_1_10.avi                                          # original video
|-- 24209_1_10_fps5000-frames                               # original frames
|-- 24209_1_10_fps5000-resize-frames-H260-W346              # resized frames, size: 260x346
|-- 24209_1_10_fps5000-resize-frames-H260-W346-events
|-- 24209_1_10_fps5000-resize-frames-H260-W346-rolling
...
"""


FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_path", None, "The path of the EvUnrall/Fastec dataset.")
flags.DEFINE_string("dataset", None, "The name of the dataset.")
flags.DEFINE_integer(
    "blur_accumulate_frames",
    130,
    "The number of frames to accumulate to generate blur.",
)
flags.DEFINE_integer(
    "blur_accumulate_step",
    260,
    "The number of frames to accumulate to generate blur.",
)


H = 260
W = 346


def rgb_to_lineal(matrix, gamma=2.2):
    """
    :param matrix: rgb matrix, value range is[0, 255]
    :param gamma:
    :return: linear matrix, value range is [0, 1]
    """
    matrix = matrix / 255.0
    return np.power(matrix, gamma)


def lineal_to_rgb(matrix, gamma=2.2):
    """
    :param matrix: linear matrix, value range is [0, 1]
    :param gamma:
    :return: rgb matrix, value range is[0, 255]
    """
    return np.power(matrix, 1 / gamma) * 255.0


def main(args):
    set_trace()
    # main
    avi_files = sorted(listdir(FLAGS.dataset_path))
    assert FLAGS.dataset in ["EvUnrall", "Fastec"], f"{FLAGS.dataset} is not supported."

    for avi_file in avi_files:
        if (not avi_file.endswith(".avi")) and FLAGS.dataset == "EvUnrall":
            continue
        elif (
            avi_file.endswith("_fps5000-resize-frames-H260-W346")
            or avi_file.endswith("_fps5000-resize-frames-H260-W346-events")
            or avi_file.endswith("_fps5000-resize-frames-H260-W346-rolling")
            or "blur-frames-H260-W346-blur-accumulate" in avi_file
        ) and FLAGS.dataset == "Fastec":
            continue
        # get the name of the video
        name = avi_file.split(".")[0]

        resize_frames_folder = join(FLAGS.dataset_path, name + "_fps5000-resize-frames-H260-W346")
        events_folder = join(FLAGS.dataset_path, name + "_fps5000-resize-frames-H260-W346-events")
        rolling_folder = join(FLAGS.dataset_path, name + "_fps5000-resize-frames-H260-W346-rolling")
        info(f"Processing {name}:")
        resize_frames = sorted(listdir(resize_frames_folder))
        events = sorted(listdir(events_folder))
        rolling = sorted(listdir(rolling_folder))
        # check the size of the folders
        resize_size = len(resize_frames)
        event_size = len(events)
        rolling_size = len(rolling)
        info(f"  resize_frames_folder:  {resize_size}, {resize_frames[0]}, {resize_frames[-1]}")
        info(f"  events_folder:         {event_size}, {events[0]}, {events[-1]}")
        info(f"  rolling_folder:        {rolling_size}, {rolling[0]}, {rolling[-1]}")
        # f"{resize_size} != {event_size / 2 + 1}"
        assert (resize_size == event_size / 2 + 1) or (resize_size == event_size + 1)
        assert rolling_size + 260 == resize_size, f"{rolling_size} != {resize_size - 260}"

        blur_frames_folder = join(
            FLAGS.dataset_path, name + f"-blur-frames-H260-W346-blur-accumulate-{FLAGS.blur_accumulate_frames}"
        )
        makedirs(blur_frames_folder, exist_ok=True)

        info(f"  to")
        info(f"    blur_frames_folder:    {blur_frames_folder}")
        for l in range(0, rolling_size - 259, FLAGS.blur_accumulate_step):
            blur_frames = np.zeros((H, W, 3), dtype=np.float32)
            info(f"   {rolling[l]} -> {rolling[l + FLAGS.blur_accumulate_frames]}")
            left_name = rolling[l].split(".")[0]
            right_name = rolling[l + FLAGS.blur_accumulate_frames].split(".")[0]
            # making blur
            for i in range(l, l + FLAGS.blur_accumulate_frames):
                image = cv2.imread(join(rolling_folder, rolling[i]))
                blur_frames += rgb_to_lineal(image)
            blur_frames /= FLAGS.blur_accumulate_frames * 1.0
            blur_frames = lineal_to_rgb(blur_frames)
            blur_frames = blur_frames.astype(np.uint8)
            blur_path = join(blur_frames_folder, f"{left_name}-{right_name}.png")
            cv2.imwrite(blur_path, blur_frames)
            info(f"    {blur_path}")


if __name__ == "__main__":
    import pudb

    pudb.set_trace()
    app.run(main)
