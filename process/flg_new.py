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
    default="/home/2t/plane/视频划分/train",
    help="输入视频路径",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="/home/aistudio/plane/视频分类/256-frame/p",
    help="结果帧存放路径",
)
parser.add_argument("-a", "--ann", type=str, default="/home/aistudio/flg/new-label")
parser.add_argument("--bs", type=int, default=8)
args = parser.parse_args()


def main():
    flg_det = PdxDet(model_path="../model/best/flg_det/", bs=args.bs)
    for vid_name in os.listdir(args.input):
        print(vid_name)
        name = vid_name.split(".")[0]
        video = Stream(
            osp.join(args.input, vid_name),
            itv_sparse=25,
            itv_dense=25,
        )
        os.mkdir(osp.join(args.output, "p" + vid_name))

        for frame_idx, img in tqdm(video, miniters=args.bs):
            if os.exists(osp.join("/home/aistudio/flg/label", vid_name + ".xml")):
                pass
            frames, idxs, bbs = flg_det.add(img, frame_idx)
            for f, idx, bb in zip(frames, idxs, bbs):
                if len(bb) != 0:
                    bb = bb[0]  # 这个网络最多检出一个起落架
                    print(bb)
                    save_path = osp.join(
                        args.output,
                        "p" + vid_name,
                        "{}-{}.png".format(name, idx),
                    )
                    # try:
                    #     cv2.imwrite(save_path, crop(f, bb.square(256)))
                    # except:
                    #     pass
        # input("here")


if __name__ == "__main__":
    main()
