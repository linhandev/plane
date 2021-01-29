# 将文件名中的空格都换成 -
import os
import os.path as osp

in_dir = "/home/lin/Desktop/data/plane/video/"
for f in os.listdir(in_dir):
    if " " in f:
        print(f)
        newf = f.replace(" ", "-")
        os.rename(osp.join(in_dir, f), osp.join(in_dir, newf))
        # input("here")
