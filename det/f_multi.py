import os
import os.path as osp
import argparse

from tqdm import tqdm
import cv2
import paddlehub as hub

from util import to_voc

parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input", type=str, help="图片路径")
parser.add_argument("-o", "--output", type=str, help="结果xml路径")
parser.add_argument("--bs", type=int, default=10, help="推理batchsize")
parser.add_argument("--model", type=str,help="模型路径")
args = parser.parse_args()




object_detector = hub.Module(name="yolov3_resnet50_vd_coco2017")

def predict(img_data, names):
    results = object_detector.object_detection(
        images=img_data,
        use_gpu=True,
        output_dir=osp.join(args.output),
        visualization=True,
    )
    for idx, res in enumerate(results):
        print(res)
        res = res["data"]
        obj_type = []
        pos = []
        for l in res:
            obj_type.append(l["label"])
            pos.append([l['left'], l['top'], l['right'], l['bottom']])
        with open(osp.join(args.output, names[idx]+".xml"), "w") as f:
            print(to_voc(names[idx], obj_type, pos), file=f)

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
