import os.path as osp
import os
import argparse
import shutil
import multiprocessing as mp


import cv2
import paddlehub as hub
import paddlex as pdx
from paddlex.det import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input", type=str, default="/home/aistudio/data/data67498/video/train", help="视频存放路径")
parser.add_argument("-o", "--output", type=str, default="/home/aistudio/data/draw", help="结果帧存放路径")
parser.add_argument("-m", "--model", type=str, default="/home/aistudio/plane/gear/output/yolov3/epoch_20", help="起落架检测模型路径")
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--itv", type=int, default=25, help="检测抽帧间隔")
args = parser.parse_args()



# 坐标的顺序是按照crop时下标的顺序，坐标第一个就是下标第一维，cv2里面的应该和这个是反的

def toint(l):
    return [int(x) for x in l]


def crop(img, p, mode="max"):
    if mode == "max":
        return img[p[0]:p[2], p[1]:p[3], :]
    elif mode == "length":
        p = toint([p[0], p[1], p[0]+p[2], p[1]+p[3]])
        return crop(img, p)

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def pinbb(p, bb):
    if bb[0] <= p[0] <= bb[2] and bb[1] <= p[1] <= bb[3]:
        return True
    return False


def dpoint(img, p, color="R"):
    if color == "R":
        color = (0, 0, 255)
    elif color == "G":
        color = (0, 255, 0)
    elif color == "B":
        color = (255, 0, 0)
    cv2.circle(img, (p[1],p[0]), 1, color, 4)


def dbb(img, b, color="R"):
    ymin, xmin, ymax, xmax = b
    lines = [
        [(xmin, ymin), (xmin, ymax)],
        [(xmax, ymin), (xmax, ymax)],
        [(xmin, ymin), (xmax, ymin)],
        [(xmin, ymax), (xmax, ymax)]
    ]
    if color == "R":
        color = (0, 0, 255)
    elif color == "G":
        color = (0, 255, 0)
    elif color == "B":
        color = (255, 0, 0)

    for l in lines:
        cv2.line(img, l[0], l[1], color, 2)

def writer(image, name, flg, people):
    if len(flg) == 0:
        return
    g = flg[0]["bbox"]
    g = toint([g[1], g[0], g[3], g[2]]) # 起落架范围
    gc = toint([g[0]+g[2]/2, g[1]+g[3]/2]) # 起落架中心
    r = [2, 3] # HWC,纵横放大几倍
    gr = toint([gc[0]-g[2]*r[0]/2, gc[1]-g[3]*r[1]/2, gc[0]+g[2]*r[0]/2, gc[1]+g[3]*r[1]/2, ]) # 一定倍数区域
    l = 128 # 以gc为中心，围一个2l边长的正方形
    gs = [gc[0]-l, gc[1]-l, gc[0]+l, gc[1]+l]
    g[2] = g[0] + g[2]
    g[3] = g[1] + g[3]

    dpoint(image, gc, "R")
    dbb(image, g)
    dbb(image, gr, "B")
    dbb(image, gs,"G")

    for p in people:
        if p['label'] != "person":
            continue
        p = toint([p['top'], p['left'], p['bottom'], p['right']])
        pc = toint([(p[0]+p[2])/2, (p[1]+p[3])/2])
        dpoint(image, pc, "G")
        dbb(image, p, "G")

    cv2.imwrite(osp.join(args.output, "draw", name), image)

def reader(image_q, vid_names):
     for vid_name in vid_names:
        print("processing {}".format(vid_name))
        vidcap = cv2.VideoCapture(osp.join(args.input, vid_name))
        idx = 0

        vid_name = vid_name.split(".")[0]
        os.mkdir(osp.join(args.output, "draw", vid_name))
        images = []
        names =  []
        while True:
            print(idx)
            vidcap.set(1, idx)
            success, image = vidcap.read()
            
            if not success:
                image_q.put([images, names])
                print("put, image qsize", image_q.qsize())
                break

            if image is not None:
                images.append(image)
                names.append(osp.join(vid_name, vid_name + "-" + str(idx).zfill(6)+".png"))
            else:
                print("None image", idx)
            

            if len(names) == args.bs:
                image_q.put([images, names])
                print("put, image qsize", image_q.qsize())
                names = []
                images = []

            
            idx += args.itv

            # shutil.move(osp.join(args.output, "draw", vid_name), osp.join(args.output, "draw-fin"))

people_det = hub.Module(name="yolov3_resnet50_vd_coco2017")
flg_det = pdx.load_model(args.model)
transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

def main(args):
    # mp.set_start_method('spawn')
    reader_q = mp.Manager().Queue(10)
    reader_num = 4
    names = os.listdir(args.input)
    names_chunk = [[] for _ in range(reader_num)]
    for idx in range(len(names)):
        names_chunk[idx%reader_num].append(names[idx])

    readers = [mp.Process(target=reader, args=(reader_q, names_chunk[idx])) for idx in range(reader_num)]
    for reader in readers:
        reader.start()
    

    writer_q= mp.Manager().Queue(10)
    writer_num = 4
    for idx in range(writer_num):
        writers = [mp.Process(target=writer, args=(writer_q)) for idx in range(writer)]
    for writer in writers:
        writer.start()

    print("finish loading")
    while True:
        print("image queue qsize", image_q.qsize())
        images, names = image_q.get()
        
        print("doing inference")
        flgs = flg_det.batch_predict(images, transforms=transforms)
        people = people_det.object_detection(images=images, use_gpu=True, visualization=False)
        for idx in range(len(names)):
            draw(images[idx], names[idx], flgs[idx], people[idx]['data'])
        print("finish inference")
    
    for reader in readers:
        reader.join()
    for writer in writers:
        writer.join()

if __name__ == "__main__":
    main(args)
