import os
import os.path as osp
import argparse

import cv2

from util.util import xml2bb, crop, dbb

parser = argparse.ArgumentParser(description="")
parser.add_argument("--img_path", type=str, default="/home/aistudio/plane/人检测/Images", help="图片路径")
parser.add_argument("--ann_path", type=str, default="/home/aistudio/plane/人检测/Annotations/", help="标注路径")
parser.add_argument("-o", "--output", type=str, default="/home/aistudio/plane/temp", help="输出路径")
args = parser.parse_args()


def main():
    for f in sorted(os.listdir(args.img_path)):
        name = f.split(".")[0]
        bbs = xml2bb(osp.join(args.ann_path, name + ".xml"))
        bbs.sort(key=lambda b: (b.wmin, b.hmin))
        img = cv2.imread(osp.join(args.img_path, name + ".png"))
        for idx, bb in enumerate(bbs):
            bbs = bb.square(64)
            if not bb.spill() and not bbs.spill() and not bbs < (64, 64):
                print(bbs)
                patch = crop(img, bbs)
                assert patch.shape == (64, 64, 3), print(patch.shape)
                cv2.imwrite(osp.join(args.output, "{}-{}.png".format(name, idx)), patch)
            else:
                print("error", bbs)


if __name__ == "__main__":
    main()
