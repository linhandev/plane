import os
import os.path as osp

path = "/home/lin/Desktop/data/plane/flg/ann/train"
for old in os.listdir(path):
    new = old.split("-")
    new = new[0] + "-" + new[1] + "_" + new[2].zfill(10)
    os.rename(osp.join(path, old), osp.join(path, new))
    print(new)
