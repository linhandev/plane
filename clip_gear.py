import os.path as osp
import os
import argparse

from tqdm import tqdm
import cv2
import paddlehub as hub
import paddlex as pdx
from paddlex.det import transforms


parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input", type=str, default="/home/aistudio/test/video", help="视频存放路径")
parser.add_argument("-o", "--output", type=str, default="/home/aistudio/test/frame", help="结果帧存放路径")
parser.add_argument("-m", "--model", type=str, default="/home/aistudio/pdx/output/yolov3/best_model", help="起落架检测模型路径")
args = parser.parse_args()


people_det = hub.Module(name="yolov3_resnet50_vd_coco2017")

flg_det = pdx.load_model(args.model)
transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

def toint(l):
    return [int(x) for x in l]

def crop(img, p, mode="max"):
    if mode == "max":
        return img[p[0]:p[2], p[1]:p[3], :]
    elif mode == "length":
        p = toint([p[0], p[1], p[0]+p[2], p[1]+p[3]])
        return crop(img, p)

for vid_name in tqdm(os.listdir(args.input)):
    print("processing {}".format(vid_name))

    vidcap = cv2.VideoCapture(osp.join(args.input, vid_name))
    idx = 0
    while True:
        vidcap.set(1, idx)
        success, image = vidcap.read()
        if not success:
            break
        
        flg = flg_det.predict(image, transforms=transforms)
        print(flg)
        if len(flg) == 0:
            idx += 25
            continue
        
        print(flg[0])
        g = flg[0]["bbox"]
        g = toint([g[1], g[0], g[3], g[2]])
        print(g)
        patch = crop(image, g)
        cv2.imwrite("/home/aistudio/test/frame/{}-ldg.png".format(idx), patch)

        idx += 25




