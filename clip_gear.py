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

# 坐标的顺序是按照crop时下标的顺序，坐标第一个就是下标第一维，cv2里面的应该和这个是反的
# 

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
    if bb[0] <= p[0] <= bb[3] and bb[1] <= p[1] <= bb[3]:
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



def main():
    for vid_name in tqdm(os.listdir(args.input)):
        print("processing {}".format(vid_name))
        vidcap = cv2.VideoCapture(osp.join(args.input, vid_name))
        idx = 0
        while True:
            vidcap.set(1, idx)
            print("----")
            print(idx)
            success, image = vidcap.read()
            if not success:
                break
            
            flg = flg_det.predict(image, transforms=transforms)
            if len(flg) == 0:
                idx += 25
                continue
            
            g = flg[0]["bbox"] # 起落架位置
            g = toint([g[1], g[0], g[3], g[2]]) 
            gc = toint([g[0]+g[2]/2, g[1]+g[3]/2]) # 起落架中心
            dpoint(image, gc, "R")
            r = [2, 3] # HWC,纵横放大几倍
            gr = toint([gc[0]-g[2]*r[0]/2, gc[1]-g[3]*r[1]/2, gc[0]+g[2]*r[0]/2, gc[1]+g[3]*r[1]/2, ])
            cv2.imwrite("/home/aistudio/test/frame/{}-gr.png".format(idx), crop(image, gr))
            patch = crop(image, g, "length")
            cv2.imwrite("/home/aistudio/test/frame/{}-ldg.png".format(idx), patch)

            g[2] = g[0] + g[2]
            g[3] = g[1] + g[3]

            people = people_det.object_detection(images=[image], use_gpu=True)[0]['data']
            for pidx, p in enumerate(people):
                if p['label'] != "person":
                    continue
                p = toint([p['top'], p['left'], p['bottom'], p['right']])
                pc = toint([(p[0]+p[2])/2, (p[1]+p[3])/2])
                dpoint(image, pc, "G")

                print("++")
                print(pidx)
                print(dist(pc, gc))
                print(pinbb(pc, g))

                cv2.imwrite("/home/aistudio/test/frame/{}-p-{}.png".format(idx, pidx), crop(image, p))

            cv2.imwrite("/home/aistudio/test/frame/{}.png".format(idx), image)

            input("here")
            idx += 25

if __name__ == "__main__":
    main()


