import os
import os.path as osp

import numpy as np
import cv2

from util.model import HumanClas

folder = "/home/aistudio/plane/弯腰分类-大/p/"
imgs = []
for name in os.listdir(folder):
    img = cv2.imread(osp.join(folder, name)).astype("float32")
    imgs.append(img)

model = HumanClas()
model.load_weight()
print(model.predict(imgs, batch_size=64))
