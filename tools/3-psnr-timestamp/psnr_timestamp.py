from os import listdir, makedirs
from os.path import exists, join

import cv2
import numpy as np
from absl import app, flags, logging


def psnr(x, y):
    d = x - y
    mse = np.mean(d * d) + 1e-6
    val = -10 * np.log10(mse)
    return val


def main(args):
    root = "ROOT_PATH_TO_YOUR_DATASET"
    psnr_list = [0 for _ in range(9)]
    count_list = [0 for _ in range(9)]
    for video in listdir(root):
        folder = join(root, video)
        file_list = sorted(listdir(folder))
        pred_list = []
        gt_list = []
        for f in file_list:
            if "_gss_pred.png" in f:
                pred_list.append(f)
            elif "_gss_gt.png" in f:
                gt_list.append(f)
        for pred, gt in zip(pred_list, gt_list):
            index = int(pred.split("_")[1])
            index_2 = int(gt.split("_")[1])
            assert index == index_2
            pred_image = cv2.imread(join(folder, pred)) / 255.0
            gt_image = cv2.imread(join(folder, gt)) / 255.0

            psnr_val = psnr(pred_image, gt_image)
            psnr_list[index] += psnr_val
            count_list[index] += 1
    print(f"psnr: {psnr_list}")
    print(f"count: {count_list}")
    print(f"index, psnr")
    for i in range(9):
        psnr_list[i] /= count_list[i]
        print(f"{psnr_list[i]}")
    print(f"average psnr: {sum(psnr_list) / 9}")


if __name__ == "__main__":
    app.run(main)
