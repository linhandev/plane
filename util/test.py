import os

import cv2

from util import Stream


vid_dir = "/home/aistudio/plane/vid-split/train/"
for name in os.listdir(vid_dir):
    vid = Stream(os.path.join(vid_dir, name))
    for img in vid:
        cv2.imshow("img", img)
        cv2.waitKey(2)
