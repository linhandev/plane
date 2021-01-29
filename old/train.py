import os
import os.path as osp

import cv2
import paddle
import paddle.vision.transforms as vt



class HumanClasDataset(paddle.io.Dataset):
    def __init__(self, mode='train'):
        super(HumanClasDataset, self).__init__()
        self.data_path = "/home/aistudio/plane/bend/"
        ps = os.listdir(osp.join(self.data_path, "p"))
        ns = os.listdir(osp.join(self.data_path, "n"))
        ps.sort()
        ns.sort()
        ps = [osp.join("p", n) for n in ps]
        ns = [osp.join("n", n) for n in ns]
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
        self.transform = vt.Compose([vt.ToTensor()])

    def __getitem__(self, index):
        data = cv2.imread(osp.join(self.data_path, self.data[index][0]))
        data = self.transform(data)
        label = self.data[index][1]
        return data, label

    def __len__(self):
        return len(self.data)




from paddle.nn import Conv2D, BatchNorm2D, ReLU, Softmax, MaxPool2D, Flatten, Linear


ClasModel = paddle.nn.Sequential(
    Conv2D(3, 6, (3,3)),
    BatchNorm2D(6),
    ReLU(),

    Conv2D(6, 6, (3,3)),
    BatchNorm2D(6),
    ReLU(),
    MaxPool2D((2,2)),

    Conv2D(6, 12, (3,3)),
    BatchNorm2D(12),
    ReLU(),

    Conv2D(12, 12, (3,3)),
    BatchNorm2D(12),
    ReLU(),
    MaxPool2D((2,2)),

    Conv2D(12, 8, (3,3)),
    BatchNorm2D(8),
    ReLU(),

    Conv2D(8, 8, (3,3)),
    BatchNorm2D(8),
    ReLU(),
    MaxPool2D((2,2)),

    Flatten(),
    Linear(128, 128),
    ReLU(),
    Linear(128, 32),
    ReLU(),
    Linear(32, 2),
    Softmax()
)

train_dataset = HumanClasDataset(mode="train")
eval_dataset = HumanClasDataset(mode="eval")

# train_loader = paddle.io.DataLoader(train_dataset, batch_size=1, shuffle=True)

model = paddle.Model(ClasModel)
model.prepare(paddle.optimizer.Adam(parameters=ClasModel.parameters()),
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy()
)
model.fit(train_dataset,
        batch_size=32,
        epochs=10,
        verbose=2
)
paddle.save(ClasModel.static_dict(), "human.pdparams")

model.evaluate(eval_dataset, verbose=1)

img = cv2.imread("/home/aistudio/plane/bend/p/15351-撤轮挡-0_2937.png")
print(model.predict(img))
