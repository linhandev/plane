import os
import os.path as osp
import random
import shutil

source = "/home/aistudio/plane/gs/"
dst = "/home/aistudio/plane/temp/"

names = random.sample(os.listdir(source), 400)
print(names)
for n in names:
    shutil.copyfile(osp.join(source, n), osp.join(dst, n))
