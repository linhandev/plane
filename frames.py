import cv2
import os.path as osp
import os
from tqdm import tqdm

for vid_name in tqdm(os.listdir("./data")):
    # vid_name = "Q11602_上轮挡.mp4"
    vidcap = cv2.VideoCapture(osp.join("data", vid_name))
    success, image = vidcap.read()
    count = 0
    while success:
        if count % 20 == 0:
            cv2.imwrite(
                osp.join("frames", vid_name.split(".")[0] + str(count) + ".png"), image
            )  # save frame as JPEG file
        success, image = vidcap.read()
        print("Read a new frame: ", success)
        count += 1
