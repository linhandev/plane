import os.path as osp
import os
import argparse

from tqdm import tqdm
import cv2
import paddlex as pdx
from paddlex.det import transforms


parser = argparse.ArgumentParser(description="")
parser.add_argument("-m","--model", default="/home/aistudio/pdx/output/yolov3/best_model", type=str, help="模型路径")
parser.add_argument("-i", "--input", default="/home/aistudio/data/flg/video", type=str, help="视频存放路径")
parser.add_argument("-o", "--output", default="/home/aistudio/data/flg/clip", type=str, help="结果帧存放路径")
parser.add_argument("--bs", type=int, default=10, help="推理batchsize")
args = parser.parse_args()


model = pdx.load_model(args.model)
transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

for vid_name in tqdm(os.listdir(args.input)):
    print("processing {}".format(vid_name))

    vidcap = cv2.VideoCapture(osp.join(args.input, vid_name))
    vid_name = vid_name.split(".")[0]
    idx = 0
    success = True
    while success:
        success, image = vidcap.read()

        print(idx)
        flg = model.predict(image, transforms=transforms)
        print(flg)
        if len(flg) == 0:
            idx += 25
            continue





        input("here")
