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
parser.add_argument("-o", "--output", type=str, default="/home/aistudio/plane/gs", help="结果帧存放路径")
parser.add_argument("-a", "--ann", type=str, default="/home/aistudio/plane/time/all")
args = parser.parse_args()


def main():
    flg_det = PdxDet(model_path="../model/best/flg_det/", bs=8)
    for vid_name in os.listdir(args.input):
        print(vid_name)
        frame_data = []
        names = []
        name = vid_name.split(".")[0]
        video = Stream(osp.join(args.input, vid_name), osp.join(args.ann, name + ".txt"), toi_only=True)

        for frame_idx, img in tqdm(video):
            # print(idx)

            # # 第一种方式推理类负责组batch
            frames, idxs, bbs = flg_det.add(img, frame_idx)
            # print(bbs)
            for f, idx, bb in zip(frames, idxs, bbs):
                if len(bb) != 0:
                    bb = bb[0]  # 这个网络最多检出一个起落架
                    save_path = osp.join(args.output, "{}-{}.png".format(name, idx))
                    try:
                        cv2.imwrite(save_path, crop(f, bb.square(256)))
                    except:
                        pass
                    # cv2.waitKey()

                    # cv2.imshow(n, crop(f, bb.square(64)))
                    # cv2.waitKey()
                    #
                    # cv2.imshow(n, crop(f, bb.region([4, 8])))
                    # cv2.waitKey()

            # 第二种方式, batch_predict, 需要注意必须清空list
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
            # bb = flg_det.predict(img)
            # if len(bb) != 0:
            #     bb = bb[0]
            #     cv2.imshow("img", crop(img, bb))
            #     cv2.waitKey()


if __name__ == "__main__":
    main()
