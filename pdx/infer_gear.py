import os
import os.path as osp

import paddlex as pdx
from paddlex.det import transforms as det_transforms
from paddlex.cls import transforms as cls_transforms
import cv2
from tqdm import tqdm
import numpy as np


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


cls_model = pdx.load_model('output/gear_clas/')
cls_trans = cls_transforms.Compose([
    # cls_transforms.Resize(),
    cls_transforms.Normalize()
])

det_model = pdx.load_model('output/gear_det/')
det_trans = det_transforms.Compose([
    det_transforms.Resize(),
    det_transforms.Normalize()
])

vid_dir = "/home/aistudio/plane/vid-split/train/"
itv = 5
bs = 4

for vid_name in tqdm(os.listdir(vid_dir)):
    print(vid_name)
    vidcap = cv2.VideoCapture(osp.join(vid_dir, vid_name))

    frame_data = []
    index = []
    frame_idx = 0
    success = True
    count = 0

    while success:
        success, image = vidcap.read()
        frame_idx += itv
        print(frame_idx)
        if success and image is not None:
            frame_data.append(image)
            print(frame_data[-1].shape)
            print("length", len(frame_data))
            if len(frame_data) == bs:

                flgs = det_model.batch_predict(frame_data, transforms=det_trans)

                flg_data = []
                for idx, flg in enumerate(flgs):
                    if len(flg) == 0:
                        continue
                    img = frame_data[idx]
                    g = flg[0]["bbox"]
                    g = toint([g[1], g[0], g[3], g[2]]) # 起落架范围
                    gc = toint([g[0]+g[2]/2, g[1]+g[3]/2]) # 起落架中心
                    l = 128 # 以gc为中心，围一个2l边长的正方形
                    gs = [gc[0]-l, gc[1]-l, gc[0]+l, gc[1]+l]
                    img = crop(img, gs)
                    flg_data.append(img)


                print("+++", len(flg_data))
                print(len(flg_data))
                if len(flg_data) != 0:
                    action_res = cls_model.batch_predict(flg_data, transforms=cls_trans)
                    for idx in range(len(flg_data)):
                        img = flg_data[idx]
                        r = action_res[idx]

                        if r[0]['category'] == 'p':
                            img[32:64,32:64, :] = [0, 255, 0]
                        else:
                            img[32:64,32:64, :] = [0, 0, 255]
                        cv2.imwrite("/home/aistudio/plane/gear-square/val/" + vid_name + "-" + str(count) + '.png', img)
                        count += 1
                flg_data = []
                frame_data = []
