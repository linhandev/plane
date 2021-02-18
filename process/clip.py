import random

import os
import os.path as osp

vid_dir = "/home/aistudio/plane/视频划分/val"
out_dir = "/home/aistudio/plane/视频分类/1920/n/"
ann_dir = "/home/aistudio/plane/时间标注/all"


for vid_name in os.listdir(vid_dir):
    print(vid_name)
    name = vid_name.split(".")[0]
    with open(osp.join(ann_dir, name + ".txt")) as f:
        try:
            info = f.read().split(" ")
            type = info.pop()
            type = type.rstrip("\n")
            # if type == "无":
            #     continue
            # print(info)
            info[0] = info[0].lstrip("\ufeff")
            info = [int(x) for x in info]
            print(info)
            # cmd = "ffmpeg -ss {} -i {} -c copy -t {} {}".format(
            #     info[0],
            #     osp.join(vid_dir, vid_name),
            #     info[1] - info[0],
            #     osp.join(out_dir, "p" + vid_name),
            # )
            cmd = "ffmpeg -ss {} -i {} -c copy -t {} {}".format(
                info[1] - 40 + int(10 * random.random()),
                osp.join(vid_dir, vid_name),
                10 + int(10 * random.random()),
                osp.join(out_dir, "n-" + vid_name),
            )

            print(cmd)
            os.system(cmd)
            # input("here")
        except:
            print(vid_name, "exception")
