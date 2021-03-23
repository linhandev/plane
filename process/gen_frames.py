import os
import os.path as osp

import cv2

from util.util import Stream

vid_path = "/home/aistudio/plane/视频划分/new-train"
img_path = "/home/aistudio/plane/temp"

for vid_name in os.listdir(vid_path):
    video = Stream(
        osp.join(vid_path, vid_name),
        itv_dense=25,
    )
    vid_name = osp.splitext(vid_name)[0]
    print(vid_name)
    # input("hre")
    for idx, frame in video:
        cv2.imwrite(osp.join(img_path, f"{vid_name}-{str(idx).zfill(5)}.png"), frame)
