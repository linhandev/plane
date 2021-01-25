import os
import os.path as osp

import cv2
import paddlex as pdx
from paddlex.det import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

vid_dir = "/home/aistudio/plane/vid-split/train"
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



flg_det = pdx.load_model("./model/new_flg/")
flg_trans = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

people_det = pdx.load_model("./model/person/50")
people_trans = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

def sort_key(p):
    return (int(p['bbox'][0]), int(p['bbox'][1]))


for vid_name in tqdm(os.listdir(vid_dir)):
    print(vid_name)

    vidcap = cv2.VideoCapture(osp.join(vid_dir, vid_name))
    vid_name = vid_name.split(".")[0]
    # os.mkdir(osp.join(out_dir, vid_name))
    with open(osp.join(time_dir, vid_name + ".txt"), "r") as f:
        times = f.read()
    s, e, action = times.split(" ")

    frame_data = []
    index = []
    for frame_idx in range(int(s)*25, int(e)*25, itv):
        vidcap.set(1, frame_idx)
        success, image = vidcap.read()
        if success and image is not None:
            frame_data.append(image)
            index.append(frame_idx)
            if len(frame_data) == bs or frame_idx + itv >= int(e)*25:
                print("length", len(frame_data))
                flgs = flg_det.batch_predict(frame_data, transforms=flg_trans)
                print(len(flgs))

                flg_data = []
                for idx, flg in enumerate(flgs):
                    img = frame_data[idx]
                    g = flg[0]["bbox"]
                    # print(g)
                    g = toint([g[1], g[0], g[3], g[2]]) # 起落架范围
                    gc = toint([g[0]+g[2]/2, g[1]+g[3]/2]) # 起落架中心
                    l = 128 # 以gc为中心，围一个2l边长的正方形
                    gs = [gc[0]-l, gc[1]-l, gc[0]+l, gc[1]+l]
                    img = crop(img, gs)
                    # cv2.imshow("img", img)
                    # cv2.waitKey()

                    flg_data.append(img)

                people = people_det.batch_predict(flg_data, transforms=people_trans)
                for fidx, persons in enumerate(people):
                    persons.sort(key=sort_key)
                    print(persons)
                    if len(persons) == 0:
                        continue
                    for pidx, p in enumerate(persons):
                        p = p["bbox"]
                        # print(p)
                        p = [p[1], p[0], p[3], p[2]]
                        pc = toint([p[0]+p[2]/2, p[1]+p[3]/2])
                        p = toint([p[0], p[1], p[0]+p[2], p[1]+p[3]])
                        l = 32
                        ps = [pc[0]-l, pc[1]-l, pc[0]+l, pc[1]+l]
                        # print(p)

                        # cv2.imshow("img", flg_data[fidx])
                        # cv2.waitKey()
                        #
                        # cv2.imshow("img", crop(flg_data[fidx], p))
                        # cv2.waitKey()
                        #
                        # cv2.imshow("img", crop(flg_data[fidx], ps))
                        # cv2.waitKey()
                        img = crop(flg_data[fidx], ps)
                        if ps[2]-ps[0] < 64 or ps[3]-ps[1]<64:
                            continue
                        print(img.shape)
                        if img.shape[0] != 64 or img.shape[1] != 64:
                            continue

                        try:
                            cv2.imwrite("/home/aistudio/plane/ps/{}-{}_{}.png".format(vid_name, pidx, index[fidx]), img)
                        except:
                            print(img)
                            print(p)

                frame_data = []
                index = []
