#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/3 10:34
from os import makedirs
from os.path import dirname, join

import cv2
import numpy as np
from absl.testing import absltest
from egrsdb.utils.flow_viz import flow_to_image

class FLowVisualizeImageTest(absltest.TestCase):
    def setUp(self):
        self.test_folder = join(dirname(__file__), "testdata", "flow_to_image")
        makedirs(self.test_folder, exist_ok=True)

    def test_flow_to_image(self):
        H = 500
        W = 500
        grid = np.zeros((H, W, 2), dtype=np.float32)
        for x in range(H):
            for y in range(W):
                dx, dy = x - H // 2, y - W // 2
                r = np.sqrt(dx * dx + dy * dy)
                if r <= H // 2:
                    grid[x, y, 0] = dx
                    grid[x, y, 1] = dy
        flow_image_n = flow_to_image(grid, convert_to_bgr=True, normalize=True)
        cv2.imwrite(join(self.test_folder, "flow_image_normalized.png"), flow_image_n)
        flow_image_wo_n = flow_to_image(grid, convert_to_bgr=True, normalize=False)
        cv2.imwrite(join(self.test_folder, "flow_image_without_normalized.png"), flow_image_wo_n)


if __name__ == "__main__":
    absltest.main()
