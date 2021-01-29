import os
import os.path as osp
import argparse

from tqdm import tqdm
import cv2
import paddlex as pdx
from paddlex.det import transforms

from util import to_voc

parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input", type=str, help="图片路径")
parser.add_argument("-o", "--output", type=str, help="结果xml路径")
parser.add_argument("--bs", type=int, default=10, help="推理batchsize")
parser.add_argument("--model", type=str,help="模型路径")
args = parser.parse_args()

model = pdx.load_model(args.model)
transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

def predict(img_data, names):
    results = model.batch_predict(img_data, transforms=transforms)
    for idx in range(len(results)):
        print(results[idx])
        try:
            r=results[idx][0]['bbox']
            with open(osp.join(args.output, names[idx]+".xml"), "w") as f:
                print(to_voc(names[idx], ["前起落架"], [[r[0], r[1], r[0]+r[2], r[1]+r[3]]]), file=f)
        except IndexError:
            with open(osp.join(args.output, names[idx]+".xml"), "w") as f:
                print(to_voc(names[idx]), file=f)
    # input("here")


img_data = []
names = []
for f in tqdm(os.listdir(args.input)):
  print(f)
  img_data.append(cv2.imread(osp.join(args.input, f)))
  names.append(f.split(".")[0])
  if len(names) == args.bs:
    predict(img_data, names)
    img_data=[]
    names=[]
