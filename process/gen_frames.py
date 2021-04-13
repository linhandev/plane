import os
import os.path as osp

from tqdm import tqdm
import cv2

from util.util import Stream

vid_path = "/home/2t/plane/视频划分/train"
img_path = "/home/aistudio/plane/temp"

for vid_name in tqdm(os.listdir(vid_path)):
    video = Stream(
        osp.join(vid_path, vid_name),
        itv_dense=12,
    )
    vid_name = osp.splitext(vid_name)[0]
    print(vid_name)
    for idx, frame in video:
        cv2.imwrite(osp.join(img_path, f"{vid_name}-{str(idx).zfill(5)}.png"), frame)
    # input("here")
