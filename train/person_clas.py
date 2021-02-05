import os
import os.path as osp

import cv2
import paddle
import paddle.vision.transforms as vt
from paddle.nn import Conv2D, BatchNorm2D, ReLU, Softmax, MaxPool2D, Flatten, Linear
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from util.model import HumanClas

paddle.disable_static()
data_dir = "/home/aistudio/plane/弯腰分类-大"


class HumanClasDataset(paddle.io.Dataset):
    def __init__(self, mode="train"):
        super(HumanClasDataset, self).__init__()
        self.data_path = data_dir
        ps = os.listdir(osp.join(self.data_path, "p"))
        ns = os.listdir(osp.join(self.data_path, "n"))
        ps.sort()
        ns.sort()
        ps = [osp.join("p", x) for x in ps]
        ns = [osp.join("n", x) for x in ns]
        data = []
        if mode == "train":
            for idx in range(int(len(ps) * 0.8)):
                data.append([ps[idx], 1])
            for idx in range(int(len(ns) * 0.8)):
                data.append([ns[idx], 0])
        else:
            for idx in range(int(len(ps) * 0.8), len(ps)):
                data.append([ps[idx], 1])
            for idx in range(int(len(ns) * 0.8), len(ns)):
                data.append([ns[idx], 0])
        self.data = data
        self.transform = vt.Compose(
            [
                # vt.ColorJitter(0.1, 0.1, 0.1, 0.1),
                # # vt.RandomRotation(10),
                # vt.RandomHorizontalFlip(),
                # vt.ColorJitter(),
                vt.Resize(64),
                vt.ToTensor(),
            ]
        )  # TODO: 研究合适的数据增强策略

    def __getitem__(self, index):
        # data = cv2.imread(osp.join(self.data_path, self.data[index][0]))
        data = Image.open(osp.join(self.data_path, self.data[index][0]))
        data = self.transform(data)
        label = self.data[index][1]
        return data, label

    def __len__(self):
        return len(self.data)


train_dataset = HumanClasDataset(mode="train")
eval_dataset = HumanClasDataset(mode="eval")


model = HumanClas().model
model.fit(
    train_dataset,
    eval_dataset,
    batch_size=64,
    epochs=100,
    eval_freq=10,
    save_dir="../model/ckpt/person_clas",
    save_freq=10,
    shuffle=True,
    # num_workers=8,
    verbose=1,
)
model.save("../model/best/person_clas/person_clas", training=True)

model.evaluate(eval_dataset, verbose=1)
