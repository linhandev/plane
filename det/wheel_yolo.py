import os.path as osp
import os
import argparse

from tqdm import tqdm
import cv2
import paddlex as pdx
from paddlex.det import transforms

model = pdx.load_model('output/yolov3/epoch_40')

transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input", type=str, help="视频存放路径")
parser.add_argument("-o", "--output", type=str, help="结果视频存放路径")
parser.add_argument("--interval", type=int, default=10, help="间隔多少帧推理一次")
parser.add_argument("--bs", type=int, default=10, help="推理batchsize")
args = parser.parse_args()


for vid_name in os.listdir(args.input):
    if not osp.exists(osp.join(args.output, vid_name)):
        os.makedirs(osp.join(args.output, vid_name))

    print("processing {}".format(vid_name))

    vidcap = cv2.VideoCapture(osp.join(args.input, vid_name))
    vidname = vidname.split(".")[0]
    success, image = vidcap.read()
    count = 0
    img_data = []
    names = []
    while success:
        if count % args.interval == 0:
            img_data.append(image)
            names.append(vidname + str(count) + ".png")
            if len(img_data) == args.bs:
                result = model.batch_predict(img_data, transforms=transforms)
                print(result)
                input("here")
                # pdx.det.visualize(image_name, result, threshold=0.001, save_dir='./output/plane_lg')

                img_data = []
                names = []


        success, image = vidcap.read()
        count += 1
        print(count)
    input("here")
