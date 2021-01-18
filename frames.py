import os.path as osp
import os
import argparse
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input", type=str, help="视频存放路径")
parser.add_argument("-o", "--output", type=str, help="结果视频存放路径")
parser.add_argument("--interval", type=int, default=10, help="间隔多少帧推理一次")
args = parser.parse_args()


def to_frames(vid_name):
    print("++", vid_name, "start")
    vidcap = cv2.VideoCapture(osp.join(args.input, vid_name))
    vid_name = vid_name.split(".")[0]
    for idx in range(0, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), args.interval):
        vidcap.set(1, idx)
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(osp.join(args.output, "{}-{}.png".format(vid_name, idx)), image)
    vidcap.release()
    print("\t--", vid_name, "end")

with Pool(6) as p:
    print(p.map(to_frames, os.listdir(args.input)))

with ThreadPoolExecutor(max_workers = 6) as executor:
  results = executor.map(to_frames, os.listdir(args.input)))
