import os
import os.path as osp

import cv2

vid_dir = "/home/aistudio/plane/视频分类/1920/n"
frame_dir = "/home/aistudio/plane/视频分类/帧"
for name in os.listdir(vid_dir):
    print(name)
    vid = cv2.VideoCapture(osp.join(vid_dir, name))
    for _ in range(100):
        success, img = vid.read()
        if success:
            break
    print(img.shape)
    name = name.split(".")[0]
    cv2.imwrite(osp.join(frame_dir, name + ".png"), img)
    vid.release()
