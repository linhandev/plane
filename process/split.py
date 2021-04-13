import numpy as np
import cv2

from util.util import Stream

video = Stream("/home/2t/plane/视频划分/long/Q216022021.3.20", itv_sparse=1500)
print(video.fps)

prev_diff = 0
print(video.shape)
prev_frame = np.zeros(video.shape)

f = open("stamp.txt", "w")
for idx, curr_frame in video:
    curr_diff = np.abs(curr_frame - prev_frame).sum()
    print(idx, prev_diff, curr_diff)
    if curr_diff > 1000 * prev_diff:
        print(int(idx / 25), file=f, end=" ")
        print("+++++++++++++++ Starting new video")
        f.flush()
    prev_diff = curr_diff
    cv2.imshow("img", curr_frame)
    cv2.waitKey(2)
