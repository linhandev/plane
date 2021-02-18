import os
import os.path as osp

import cv2

from util.util import xml2bb, Stream, crop

vid_dir = "/home/aistudio/plane/视频分类/1920/p/"
bb_dir = "/home/aistudio/plane/视频分类/起落架bb/"
out_dir = "/home/aistudio/plane/视频分类/256-frame/p/"

for name in os.listdir(vid_dir):
    name = name.split(".")[0]
    bb_path = osp.join(bb_dir, name + ".xml")
    if os.path.exists(bb_path):
        print(name)
        bb = xml2bb(bb_path, "gear")[0]
        vid = cv2.VideoCapture(osp.join(vid_dir, name + ".mp4"))
        stream = Stream(osp.join(vid_dir, name + ".mp4"), itv_sparse=2)

        os.mkdir(osp.join(out_dir, name))

        for idx, img in stream:
            print(idx)
            # cv2.imshow("img", img)
            # cv2.waitKey()

            out_path = osp.join(out_dir, name, "{}-{}.png".format(name, idx))
            img = crop(img, bb.square(256), do_pad=True)
            # cv2.imshow("img", img)
            # cv2.waitKey()
            cv2.imwrite(out_path, img)
