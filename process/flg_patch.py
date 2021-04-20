import os
import os.path as osp
import argparse
import shutil

import cv2
from tqdm import tqdm
import numpy as np

from util.util import BB, crop, Stream, dbb, dpoint
from util.model import PdxDet, HumanClas

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="../train-video",
    help="视频存放路径",
)
parser.add_argument("-o", "--output", type=str, default="../temp", help="结果帧存放路径")
parser.add_argument("--itv", type=int, default=25, help="抽帧间隔")
parser.add_argument("--bs", type=int, default=2, help="推理bs")
args = parser.parse_args()


def main():
    # 1. 定义模型对象
    flg_det = PdxDet(model_path="../model/best/flg_det/best_model", bs=16)

    for vid_name in tqdm(os.listdir(args.input)):
        print(vid_name)
        video = Stream(
            osp.join(args.input, vid_name),
            itv_sparse=25,
            itv_dense=3,
        )
        for fidx, img in video:
            # 检测出一个batch的起落架s
            frames, infos, flgs_batch = flg_det.add(
                img,
                [fidx, vid_name, video.in_toi()[0]],
            )
            for frame, info, flgs in zip(frames, infos, flgs_batch):  # 对这些起落架中的每一个
                frame_idx, video_name, flag_in_toi = info
                print(flgs)
                if len(flgs) != 0:
                    flg = flgs[0]
                else:
                    continue
                img_path = osp.join(
                    args.output,
                    f"{osp.splitext(video_name)[0]}_{str(frame_idx).zfill(5)}_{str(flag_in_toi + 0)}.png",
                )
                print(img_path)
                # img_file = crop(f, flg.square(256))[1].tofile(img_path)
                cv2.imencode(".png", img)[1].tofile(img_path)   # windows下使用防乱码
                # cv2.imwrite(img_path, crop(frame, flg.square(256)))


if __name__ == "__main__":
    main()
