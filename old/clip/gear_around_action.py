import os
import os.path as osp

import cv2
import paddlex as pdx
from paddlex.det import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

vid_dir = "/home/aistudio/plane/vid-split/val"
time_dir = "/home/aistudio/plane/time/all"
out_dir = "/home/aistudio/plane/gs-action"
bs = 8
itv = 8

def toint(l):
    return [int(x) for x in l]


def crop(img, p, mode="max"):
    if mode == "max":
        return img[p[0]:p[2], p[1]:p[3], :]
    elif mode == "length":
        p = toint([p[0], p[1], p[0]+p[2], p[1]+p[3]])
        return crop(img, p)


def dbb(img, b, color="R"):
    ymin, xmin, ymax, xmax = b
    lines = [
        [(xmin, ymin), (xmin, ymax)],
        [(xmax, ymin), (xmax, ymax)],
        [(xmin, ymin), (xmax, ymin)],
        [(xmin, ymax), (xmax, ymax)]
    ]
    if color == "R":
        color = (0, 0, 255)
    elif color == "G":
        color = (0, 255, 0)
    elif color == "B":
        color = (255, 0, 0)

    for l in lines:
        cv2.line(img, l[0], l[1], color, 2)



def dbb(img, b, color="R"):
    ymin, xmin, ymax, xmax = b
    lines = [
        [(xmin, ymin), (xmin, ymax)],
        [(xmax, ymin), (xmax, ymax)],
        [(xmin, ymin), (xmax, ymin)],
        [(xmin, ymax), (xmax, ymax)]
    ]
    if color == "R":
        color = (0, 0, 255)
    elif color == "G":
        color = (0, 255, 0)
    elif color == "B":
        color = (255, 0, 0)

    for l in lines:
        cv2.line(img, l[0], l[1], color, 2)



flg_det = pdx.load_model("../model/new_flg/")
flg_trans = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])


def sort_key(p):
    return (int(p['bbox'][0]), int(p['bbox'][1]))


for vid_name in tqdm(os.listdir(vid_dir)):
    print(vid_name)

    vidcap = cv2.VideoCapture(osp.join(vid_dir, vid_name))
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frame_count)
    vid_length = frame_count/25
    # input("here")
    vid_name = vid_name.split(".")[0]
    # os.mkdir(osp.join(out_dir, vid_name))
    with open(osp.join(time_dir, vid_name + ".txt"), "r") as f:
        times = f.read()
    s, e, action = times.split(" ")
    s, e = int(s), int(e)
    frame_data = []
    index = []
    if s < vid_length / 2:
        r = range((e + 15)*25, (e + 20)*25, itv)
    else:
        r = range((s - 20)*25, (s - 15)*25, itv)

    for frame_idx in r:
        vidcap.set(1, frame_idx)
        success, image = vidcap.read()
        if success and image is not None:
            frame_data.append(image)
            index.append(frame_idx)
            print(len(frame_data))
            if len(frame_data) == bs or frame_idx + itv >= frame_count:
                print("length", len(frame_data))
                flgs = flg_det.batch_predict(frame_data, transforms=flg_trans)
                print(len(flgs))

                for idx, flg in enumerate(flgs):
                    if len(flg) == 0:
                        continue
                    img = frame_data[idx]
                    g = flg[0]["bbox"]
                    # print(g)
                    g = toint([g[1], g[0], g[3], g[2]]) # 起落架范围
                    gc = toint([g[0]+g[2]/2, g[1]+g[3]/2]) # 起落架中心
                    l = 128 # 以gc为中心，围一个2l边长的正方形
                    gs = [gc[0]-l, gc[1]-l, gc[0]+l, gc[1]+l]
                    img = crop(img, gs)
                    cv2.imwrite("/home/aistudio/plane/gear-square/val/n/"+vid_name+"-"+str(index[idx])+'.png', img)

                frame_data = []
                index = []
