import os
import io
import random

import paddle
from paddle.nn import functional as F
from paddle.io import Dataset
from paddle.vision.models import LeNet
import paddle.vision.transforms as vT
from paddle import nn
import cv2
import numpy as np

seq_len = 2


def create_img(num):
    assert num >= 0 and num <= 9, f"Num should be between 0 and 0, got {num}"
    img = np.zeros((28, 28, 1), np.float32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5, 23)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 1
    cv2.putText(img, str(num), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    return img


class SeqDataset(Dataset):
    def __init__(self):
        super(SeqDataset, self).__init__()
        self.seq_len = seq_len
        self.tot_num = 100
        self.transforms = vT.Compose([vT.Transpose((2, 1, 0))])

    # 每次迭代时返回数据和对应的标签
    def __getitem__(self, idx):
        nums = [random.randint(0, 9) for _ in range(self.seq_len)]
        imgs = [self.transforms(create_img(n)) for n in nums]
        # imgs = [create_img(n) for n in nums]
        imgs = np.array(imgs)
        res = np.zeros([2, 10], np.float32)
        res[0][nums[1]] = 1
        res[1][nums[0]] = 1
        return imgs, res

    # 返回整个数据集的总数
    def __len__(self):
        return self.tot_num


train_dataset = SeqDataset()

# for x, y in train_dataset:
#     print(y)
#     cv2.imshow("img", x[0])
#     cv2.waitKey()
#     cv2.imshow("img", x[1])
#     cv2.waitKey()


class SoftmaxWithCrossEntropy(paddle.nn.Layer):
    def __init__(self):
        super(SoftmaxWithCrossEntropy, self).__init__()

    def forward(self, input, label):
        s = input.shape
        s[0] *= s[1]
        del s[1]
        input = paddle.reshape(input, s)
        label = paddle.reshape(label, s)
        loss = F.softmax_with_cross_entropy(input, label, axis=-1, soft_label=True)
        return paddle.mean(loss)


class LSTMModel(paddle.nn.Layer):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.cnn = LeNet()
        self.lstm = nn.LSTM(10, 32, 2, direction="bidirect")
        self.fc = nn.Linear(64, 10)

    def forward(self, inputs):
        s = inputs.shape
        bs = s[0]
        s[1] *= s[0]
        del s[0]
        x = paddle.reshape(inputs, s)
        x = self.cnn(x)
        s = x.shape
        s[0] = int(s[0] / bs)
        s.insert(0, bs)
        x = paddle.reshape(x, s)
        x, (h, c) = self.lstm(x)
        x = self.fc(x)
        return x


model = paddle.Model(LSTMModel())
print(model.summary((4, 2, 1, 28, 28)))
model.prepare(
    paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
    SoftmaxWithCrossEntropy(),
    paddle.metric.Accuracy(),
)
model.fit(train_dataset, epochs=10000, batch_size=8, verbose=1)
