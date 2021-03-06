import os.path as osp
import os
import argparse

from tqdm import tqdm
import cv2
import paddlehub as hub


parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input", type=str, help="视频存放路径")
parser.add_argument("-o", "--output", type=str, help="结果视频存放路径")
parser.add_argument("--interval", type=int, default=10, help="间隔多少帧推理一次")
parser.add_argument("--bs", type=int, default=10, help="推理batchsize")
args = parser.parse_args()


object_detector = hub.Module(name="yolov3_resnet50_vd_coco2017")

for vid_name in os.listdir(args.input):
    if not osp.exists(osp.join(args.output, vid_name)):
        os.makedirs(osp.join(args.output, vid_name))

    print("processing {}".format(vid_name))

    vidcap = cv2.VideoCapture(osp.join(args.input, vid_name))
    success, image = vidcap.read()
    count = 0
    img_data = []
    frames = []
    while success:
        if count % args.interval == 0:
            img_data.append(image)
            frames.append(count)
            if len(img_data) == args.bs:
                result = object_detector.object_detection(
                    images=img_data,
                    use_gpu=True,
                    output_dir=osp.join(args.output, vid_name),
                    visualization=True,
                )
                for res, frame in zip(result, frames):
                    os.rename(
                        res["save_path"],
                        osp.join(args.output, vid_name, str(frame).zfill(6) + ".jpg"),
                    )
                img_data = []
                frames = []

                print(result)

        success, image = vidcap.read()
        count += 1
        print(count)
    input("here")
