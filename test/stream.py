###########################################################
# Stream Test
###########################################################
from util.util import Stream
import cv2

vid = Stream(
    "/home/aistudio/plane/vid-split/train/15251-上轮挡.mp4",
    "/home/aistudio/plane/时间标注/train/15251-上轮挡.txt",
    itv_sparse=0,
    itv_dense=5,
)
for idx, frame in vid:
    cv2.imshow("img", frame)
    cv2.waitKey()
