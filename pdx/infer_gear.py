import os

import paddlex as pdx
from paddlex.det import transforms

model = pdx.load_model('output/yolov3/epoch_40')
img_dir = "/home/aistudio/data/val"
transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

names = []
for n in os.listdir(img_dir):
    if len(n) == 8:
        result = model.batch_predict(nams, transforms=transforms)
    names = []
    print(result)
# pdx.det.visualize(image_name, result, threshold=0.001, save_dir='./output/plane_lg')
