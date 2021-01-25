import os
import os.path as osp

import paddlex as pdx
from paddlex.det import transforms

model = pdx.load_model('output/gear_clas/epoch_10')
img_dir = "/home/aistudio/data/val"
transforms = transforms.Compose([
    transforms.Normalize()
])

names = []
for n in os.listdir(img_dir):
    names.append(osp.join(img_dir, n))
    # print(names)
    if len(names) == 8:
        result = model.batch_predict(names, transforms=transforms)
        names = []
        print(result)
# pdx.det.visualize(image_name, result, threshold=0.001, save_dir='./output/plane_lg')
