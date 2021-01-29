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
    for vid_name in os.listdir(args.input):
        print(vid_name)
        frame_data = []
        names = []

        for idx, img in tqdm(enumerate(Stream(osp.join(args.input, vid_name)))):
            print(idx)

            # # 第一种方式推理类负责组batch
            # frames, names, bbs = flg_det.add(img, idx)
            # print(bbs)
            # for f, n, bb in zip(frames, names, bbs):
            #     if len(bb) != 0:
            #         bb = bb[0]  # 这个网络最多检出一个起落架
            #         cv2.imshow(n, crop(f, bb))
            #         cv2.waitKey()
            #
            #         # cv2.imshow(n, crop(f, bb.square(64)))
            #         # cv2.waitKey()
            #         #
            #         # cv2.imshow(n, crop(f, bb.region([4, 8])))
            #         # cv2.waitKey()

            # # 第二种方式, batch_predict, 需要注意必须清空list
            # frame_data.append(img)
            # names.append(str(idx))
            # if len(frame_data) == args.bs:
            #     bbs = flg_det.batch_predict(frame_data)
            #     for idx, bb in enumerate(bbs):
            #         if len(bb) != 0:
            #             bb = bb[0]
            #             cv2.imshow("img", crop(frame_data[idx], bb))
            #             cv2.waitKey()
            #     frame_data = []
            #     names = []

            # 第三种方式,单张进行推理
            bb = flg_det.predict(img)
            if len(bb) != 0:
                bb = bb[0]
                cv2.imshow("img", crop(img, bb))
                cv2.waitKey()


if __name__ == "__main__":
    main()
