import os.path as osp
import os
import argparse

from tqdm import tqdm
import cv2
import paddlex as pdx
from paddlex.det import transforms

parser = argparse.ArgumentParser(description="")
parser.add_argument("-m","--model",type=str, help="模型路径")
parser.add_argument("-i", "--input", type=str, help="视频存放路径")
parser.add_argument("-o", "--output", type=str, help="结果视频存放路径")
parser.add_argument("--interval", type=int, default=10, help="间隔多少帧推理一次")
parser.add_argument("--bs", type=int, default=10, help="推理batchsize")
args = parser.parse_args()


model = pdx.load_model(args.model)

transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

def predict(img_data, names, folder):
    results = model.batch_predict(img_data, transforms=transforms)
    for idx in range(len(results)):
        vis = pdx.det.visualize(img_data[idx], results[idx], threshold=0.8, save_dir=None)
        cv2.imwrite(osp.join(folder, names[idx]), vis)
        print(idx)

    print("--------------------------")
    print(folder)
    for res in results:
        print(res)
    # input("here")

for vid_name in tqdm(os.listdir(args.input)):
    print("processing {}".format(vid_name))

    vidcap = cv2.VideoCapture(osp.join(args.input, vid_name))

    vid_name = vid_name.split(".")[0]
    folder = osp.join(args.output, vid_name)
    if not osp.exists(folder):
        os.makedirs(folder)

    success, image = vidcap.read()
    count = 0
    img_data = []
    names = []
    while success:
        if count % args.interval == 0:
            img_data.append(image)
            names.append(str(count).zfill(6) + ".jpg")
            if len(img_data) == args.bs:
                predict(img_data, names, folder)
                img_data = []
                names = []


        success, image = vidcap.read()
        count += 1
        # print(count)

    # 如果有剩的
    predict(img_data, names, folder)
    # input("here")
