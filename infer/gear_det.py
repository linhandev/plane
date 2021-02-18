import os
import os.path as osp
import argparse

import cv2
from tqdm import tqdm
import numpy as np

from util.util import BB, crop, Stream
from util.model import PdxDet

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="/home/aistudio/plane/vid-split/train",
    help="视频存放路径",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="/home/aistudio/plane/起落架256/toi中/",
    help="结果帧存放路径",
)
parser.add_argument("-a", "--ann", type=str, default="/home/aistudio/plane/时间标注/all/")
parser.add_argument("--bs", type=int, default=8)
args = parser.parse_args()


def main():
    flg_det = PdxDet(model_path="../model/best/flg_det/", bs=args.bs)
    for vid_name in os.listdir(args.input):
        print(vid_name)
        name = vid_name.split(".")[0]
        video = Stream(
            osp.join(args.input, vid_name),
            osp.join(args.ann, name + ".txt"),
            itv_sparse=0,
            itv_dense=5,
        )

        for frame_idx, img in tqdm(video, miniters=args.bs):
            frames, idxs, bbs = flg_det.add(img, frame_idx)
            for f, idx, bb in zip(frames, idxs, bbs):
                if len(bb) != 0:
                    bb = bb[0]  # 这个网络最多检出一个起落架
                    save_path = osp.join(args.output, "{}-{}.png".format(name, idx))
                    try:
                        cv2.imwrite(save_path, crop(f, bb.square(256)))
                    except:
                        pass
        # input("here")


if __name__ == "__main__":
    main()
