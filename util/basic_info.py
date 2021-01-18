import os
import os.path as osp

import cv2

vid_dir = "/home/lin/Desktop/data/plane/video/all/"
for vid_name in os.listdir(vid_dir):
    print(vid_name, end=", ")
    video = cv2.VideoCapture(osp.join(vid_dir, vid_name))
    print(video.get(cv2.CAP_PROP_FPS), end=", ")
    print(video.get(cv2.CAP_PROP_FRAME_COUNT), end=", ")
    print(video.get(cv2.CAP_PROP_FRAME_COUNT)/video.get(cv2.CAP_PROP_FPS)/60, end=", ")
    print(video.get(cv2.CAP_PROP_FRAME_WIDTH), end=", ")
    print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
