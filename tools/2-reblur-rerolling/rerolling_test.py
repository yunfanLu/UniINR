from glob import glob
from os.path import dirname, join

import numpy as np
from absl import app


def load_events(event_folder, start, end):
    """load events from folder."""
    events = []
    event_files = sorted(glob.glob(join(event_folder, "*.npy")))
    for x in range(start, end):
        events.append(np.load(event_files[x]))
    events = np.array(events)
    return events


def main(args):
    del args
    # test root
    root = "/mnt/dev-ssd-8T/yunfanlu/workspace/dataset/3-10000fps-videos/1-Videos-EvNurall/"
    rs_blur_folder = "24209_1_33_fps5000-blur-frames-H260-W346-blur-accumulate-260"
    rs_folder = "24209_1_33_fps5000-resize-frames-H260-W346-rolling"
    gs_folder = "24209_1_33_fps5000-resize-frames-H260-W346"
    event_folder = "24209_1_33_fps5000-resize-frames-H260-W346-events"
    # test folder name
    test_folder = join(dirname(__file__), "testdata", "rerolling_test")
    # given an rolling global shutter frame and transforme it to rolling and blur.
    gs_frame_260 = join(root, gs_folder, "0000000260.png")
    rs_frame_ = join(root, rs_folder, "0000000390.png")
    rs_blur_frame_ = join(root, rs_blur_folder, "0000000260-0000000520.png")
    events = load_events(event_folder, start=0, end=520)


if __name__ == "__main__":
    app.run(main)
