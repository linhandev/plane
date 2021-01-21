import os.path as osp
import os
import argparse
import math
import shutil

from tqdm import tqdm
import cv2
import paddlehub as hub
import paddlex as pdx
from paddlex.det import transforms


parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input", type=str, default="/home/aistudio/test/video", help="视频存放路径")
parser.add_argument("-o", "--output", type=str, default="/home/aistudio/test/frame", help="结果帧存放路径")
parser.add_argument("-m", "--model", type=str, default="/home/aistudio/pdx/output/yolov3/best_model", help="起落架检测模型路径")
parser.add_argument("--itv", type=int, default=8, help="人进入起落架区域，抽帧间隔")
args = parser.parse_args()


people_det = hub.Module(name="yolov3_resnet50_vd_coco2017")

flg_det = pdx.load_model(args.model)
transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

# 坐标的顺序是按照crop时下标的顺序，坐标第一个就是下标第一维，cv2里面的应该和这个是反的

def toint(l):
    return [int(x) for x in l]


def crop(img, p, mode="max"):
    if mode == "max":
        return img[p[0]:p[2], p[1]:p[3], :]
    elif mode == "length":
        p = toint([p[0], p[1], p[0]+p[2], p[1]+p[3]])
        return crop(img, p)

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def pinbb(p, bb):
    if bb[0] <= p[0] <= bb[2] and bb[1] <= p[1] <= bb[3]:
        return True
    return False


def dpoint(img, p, color="R"):
    if color == "R":
        color = (0, 0, 255)
    elif color == "G":
        color = (0, 255, 0)
    elif color == "B":
        color = (255, 0, 0)
    cv2.circle(img, (p[1],p[0]), 1, color, 4)


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

def main():
    for vid_name in tqdm(os.listdir(args.input)):
        print("processing {}".format(vid_name))
        vidcap = cv2.VideoCapture(osp.join(args.input, vid_name))
        idx = 0
        count = 0
        savegs = False
        savegr = False
        vid_name = vid_name.split(".")[0]
        os.mkdir(osp.join(args.output, "frame", vid_name))
        while True:
            vidcap.set(1, idx)
            print(idx)
            success, image = vidcap.read()
            if not success: # 视频到头
                break

            flg = flg_det.predict(image, transforms=transforms)
            if len(flg) == 0: # 没有起落架，过
                idx += 25
                continue

            g = flg[0]["bbox"]
            g = toint([g[1], g[0], g[3], g[2]]) # 起落架范围
            gc = toint([g[0]+g[2]/2, g[1]+g[3]/2]) # 起落架中心
            r = [2, 3] # HWC,纵横放大几倍
            gr = toint([gc[0]-g[2]*r[0]/2, gc[1]-g[3]*r[1]/2, gc[0]+g[2]*r[0]/2, gc[1]+g[3]*r[1]/2, ]) # 一定倍数区域
            l = 128 # 以gc为中心，围一个2l边长的正方形
            gs = [gc[0]-l, gc[1]-l, gc[0]+l, gc[1]+l]
            g[2] = g[0] + g[2]
            g[3] = g[1] + g[3]



            if count != 0: # 前面的帧看到人在起落架周围了，直接保存
                if savegs:
                    cv2.imwrite(osp.join(args.output, "gs", "{}-{}-gs.png".format(vid_name, str(idx).zfill(6))), crop(image, gs))
                if savegr:
                    cv2.imwrite(osp.join(args.output, "gr", "{}-{}-gr.png".format(vid_name, str(idx).zfill(6))), crop(image, gr))
                count -= 1
                idx += args.itv
                continue

            savegs = savegr = False

            people = people_det.object_detection(images=[image], use_gpu=True)[0]['data']

            for pidx, p in enumerate(people):
                if p['label'] != "person":
                    continue
                p = toint([p['top'], p['left'], p['bottom'], p['right']])
                pc = toint([(p[0]+p[2])/2, (p[1]+p[3])/2])
                dpoint(image, pc, "G")
                dbb(image, p, "G")
                if pinbb(pc, gr):
                    count = math.floor(25/args.itv)
                    savegr = True

                if pinbb(pc, gs):
                    count = math.floor(25/args.itv)
                    savegs = True


            # 检测完人再乱涂乱画
            dpoint(image, gc, "R")
            dbb(image, g)
            dbb(image, gr, "B")
            dbb(image, gs,"G")
            cv2.imwrite(osp.join(args.output, "frame", vid_name, "{}-{}.png".format(vid_name, str(idx).zfill(6))), image)

            if count == 0:
                idx += 25
        shutil.move(osp.join(args.output, "frame", vid_name), osp.join(args.output, "fin"))

if __name__ == "__main__":
    main()
