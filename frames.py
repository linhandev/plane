import cv2
import os.path as osp
import os
from tqdm import tqdm

for vid_name in tqdm(os.listdir("./vid")):
    # vid_name = "Q11602_上轮挡.mp4"
    vidcap = cv2.VideoCapture(osp.join("vid", vid_name))
    success, image = vidcap.read()
    count = 0
    while success:
        if count % 250 == 0:
            cv2.imwrite(
                osp.join("frame", vid_name.split(".")[0] + str(count) + ".png"), image
            )  # save frame as JPEG file
        success, image = vidcap.read()
        print("Read a new frame: ", success)
        count += 1
