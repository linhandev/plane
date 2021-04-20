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
    default="../train-video111",
    help="视频存放路径",
)
parser.add_argument("-o", "--output", type=str, default="../temp111", help="结果帧存放路径")
parser.add_argument("-t", "--time", type=str, default="../util", help="上撤轮挡时间标注")
parser.add_argument("--itv", type=int, default=25, help="抽帧间隔")
parser.add_argument("--bs", type=int, default=2, help="推理bs")
args = parser.parse_args()


def main():
    # 1. 定义模型对象
    flg_det = PdxDet(model_path="../model/best/flg_det/best_model/", bs=2)
    person_det = PdxDet(model_path="../model/best/person_det_yolov3", bs=2, autoflush=False)
    person_clas = HumanClas(mode="predict")

    for vid_name in os.listdir(args.input):
        # print(vid_name)
        # os.mkdir(osp.join(args.output, vid_name.split(".")[0]))

        video = Stream(
            osp.join(args.input, vid_name),
            osp.join(args.time, "time_mark.CSV"),
            itv_sparse=25,
            itv_dense=3,
        )
        mem_len = 8
        history = [False for _ in range(mem_len)]
        thresh = 0.5
        # TODO: 研究tqdm需要什么方法显示总数
        # res_f = open(osp.join("H:/W_S/Graduation_Project/plane/time-out", vid_name.rstrip(".mp4") + ".txt"), "w")
        for fidx, img, flag_in_toi in video:
            # 检测出一个batch的起落架s
            frames, info, flgs_batch = flg_det.add(img, [fidx, vid_name])
            for frame, (frame_idx, vid_name), flgs in zip(frames, info, flgs_batch):  # 对这些起落架中的每一个
                if len(flgs) != 0:
                    flg = flgs[0]
                    person_det.add(crop(frame, flg.square(256)), [flg, frame_idx, frame])  # 添加到检测人的任务list中
            # print("Gears detected: ", flgs_batch)
            if len(person_det.imgs) >= person_det.bs:
                r = person_det.flush()  # 进行人像检测
                # print("People detected: ", r[2])
                for gear_square, info, people in zip(r[0], r[1], r[2]):  # 对一个batch中的每一张，每一张可能有多个人
                    flg = info[0]
                    fid = info[1]
                    f = info[2]
                    # TODO: 一个batch推理
                    has_positive = False
                    # for pid, person in enumerate(people):
                    #     patch = crop(f, flg.square(256).transpose(person).square(64))
                    #     res = person_clas.predict(patch)
                    #     if res:
                    #         has_positive = True
                    #     # dbb(f, flg.square(256).transpose(person).region([1.8, 1.8]), "G" if res else "R")
                    #     # dpoint(f, flg.square(256).transpose(person).center(), "G" if res else "R")

                    # for idx in range(mem_len - 1, 0, -1):
                    #     history[idx] = history[idx - 1]
                    # history[0] = has_positive
                    # prediction = "Positive" if np.sum(history) > mem_len * thresh else "Negative"
                    # print(has_positive)
                    # print(history, np.sum(history), prediction)
                    # print(fid, has_positive, prediction, np.sum(history), file=res_f)
                    # res_f.flush()
                    # dbb(f, flg, "B")
                    # dpoint(f, flg.center(), "B")
                    # dbb(f, flg.square(256), "B")
                    # cv2.imshow("img", f)
                    # cv2.waitKey()
                    print(osp.join(args.output, vid_name.split(".")[0], str(fid).zfill(5) + ".png"))
                    cv2.imencode(".png", crop(f, flg.square(256)))[1].tofile(osp.join(args.output,
                                                                                      vid_name.split(".")[0].split("-")[
                                                                                          0] + "_" +
                                                                                      vid_name.split(".")[0].split("-")[
                                                                                          1] + "_" + str(fid).zfill(
                                                                                          5) + "_" + str(
                                                                                          flag_in_toi + 0) + ".png"))
        # res_f.close()
        # shutil.move(osp.join(args.input, vid_name), osp.join(args.output, "finish"))


if __name__ == "__main__":
    main()
