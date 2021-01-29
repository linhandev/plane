import os
import os.path as osp
import argparse

import cv2
from tqdm import tqdm
import numpy as np

from util.util import BB, crop, Stream, PdxDet

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="/home/aistudio/plane/vid-split/train",
    help="视频存放路径",
)
parser.add_argument(
    "-o", "--output", type=str, default="/home/aistudio/plane/temp", help="结果帧存放路径"
)
parser.add_argument("--itv", type=int, default=100, help="抽帧间隔")
parser.add_argument("--bs", type=int, default=2, help="推理bs")
args = parser.parse_args()


def main():
    flg_det = PdxDet(model_path="../model/best/flg_det/", bs=4)
    person_det = PdxDet(model_path="../model/best/person_det", bs=4)
    for vid_name in os.listdir(args.input):
        print(vid_name)
        frame_data = []
        names = []

        # TODO: 研究tqdm需要什么方法显示总数
        for idx, img in tqdm(enumerate(Stream(osp.join(args.input, vid_name)))):
            frames, names, bbs = flg_det.add(img, idx)
            flgs = []
            for f, n, bb in zip(frames, names, bbs):
                if len(bb) != 0:
                    flgs.append(crop(f, bb[0].square(256)))

            if len(flgs) != 0:
                people_batch = person_det.batch_predict(flgs)
                print(people_batch)
                for idx, people in enumerate(people_batch):
                    for person in people:
                        cv2.imshow("img", crop(flgs[idx], person.square(64)))
                        cv2.waitKey()
                # input("here")


if __name__ == "__main__":
    main()
