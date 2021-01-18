import os
import argparse

import paddlex as pdx

parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input", type=str, help="图片路径")
parser.add_argument("-o", "--output", type=str, help="结果xml路径")
parser.add_argument("--bs", type=int, default=10, help="推理batchsize")
parser.add_argument("--model", type=str,help="模型路径")
args = parser.parse_args()


def to_voc(name, xmin,ymin,xmax,ymax):
    res = """<?xml version='1.0' encoding='UTF-8'?>
    <annotation>
      <filename>{}</filename>
      <object_num>1</object_num>
      <size>
        <width>1920</width>
        <height>1080</height>
      </size>
      <object>
        <name>前起落架</name>
        <bndbox>
          <xmin>{}</xmin>
          <ymin>{}</ymin>
          <xmax>{}</xmax>
          <ymax>{}</ymax>
        </bndbox>
      </object>
    </annotation>""".format("1.png", 1,2,3,4)
    return res


model = pdx.load_model(args.model)
transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

def predict(img_data, names):
    results = model.batch_predict(img_data, transforms=transforms)
    for idx in range(len(results)):
        results[idx]=r
        print(r)
        with open(osp.join(args.output, names[idx]+".xml")) as f:
            print(to_voc(names[i], ))
        input("here")


img_data = []
names = []
for f in os.listdir(args.input):
    img_data.append(cv2.imread(osp.join(args.input, f)))
    names.append(f.split(".")[0])
