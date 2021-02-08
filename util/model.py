import paddle
from paddle.nn import Conv2D, BatchNorm2D, ReLU, Softmax, MaxPool2D, Flatten, Linear
import paddlex as pdx
from paddlex.det import transforms as dT
from paddle.static import InputSpec
import numpy as np


from .util import BB


class HumanClas:
    def __init__(self, mode="train"):
        ClasModel = paddle.nn.Sequential(
            Conv2D(3, 6, (3, 3)),
            BatchNorm2D(6),
            ReLU(),
            Conv2D(6, 6, (3, 3)),
            BatchNorm2D(6),
            ReLU(),
            MaxPool2D((2, 2)),
            Conv2D(6, 12, (3, 3)),
            BatchNorm2D(12),
            ReLU(),
            Conv2D(12, 12, (3, 3)),
            BatchNorm2D(12),
            ReLU(),
            MaxPool2D((2, 2)),
            Conv2D(12, 8, (3, 3)),
            BatchNorm2D(8),
            ReLU(),
            Conv2D(8, 8, (3, 3)),
            BatchNorm2D(8),
            ReLU(),
            MaxPool2D((2, 2)),
            Flatten(),
            Linear(128, 128),
            ReLU(),
            Linear(128, 32),
            ReLU(),
            Linear(32, 2),
            Softmax(),
        )
        input = InputSpec([None, 3, 64, 64], "float32", "x")
        label = InputSpec([None, 1], "int32", "label")
        model = paddle.Model(ClasModel, inputs=input, labels=label)
        model.prepare(
            paddle.optimizer.Adam(parameters=ClasModel.parameters()),
            paddle.nn.CrossEntropyLoss(),
            paddle.metric.Accuracy(),
        )
        self.model = model

        if mode == "predict":
            self.load_weight()

    def load_weight(self, path="../model/best/person_clas/person_clas"):
        self.model.load(path)

    def predict(self, img, batch_size=None):
        print(img.shape)
        img = img.astype("float32")
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 0, 2)
        img = img[np.newaxis, :, :, :]
        img = [img]
        res = self.model.predict(img)
        print(res)
        res = res[0][0][0][0] < 0.8
        return res


class PdxDet:
    def __init__(self, model_path, bs=4, thresh=0.9, autoflush=True):
        self.model = pdx.load_model(model_path)
        self.transform = dT.Compose([dT.Resize(), dT.Normalize()])
        self.bs = bs
        self.thresh = thresh
        self.imgs = []
        self.infos = []
        self.autoflush = autoflush

    def predict(self, img):
        """推理一张图片.

        Parameters
        ----------
        img : np.ndarray
            需要推理的一张图片.

        Returns
        -------
        list
            推理的结果，包含0到多个BB的列表.

        """
        bbs = self.model.predict(img, transforms=self.transform)
        res = []
        if len(bbs) != 0:
            for bb in bbs:
                res.append(BB(bb["bbox"], type="pdx"))
        return res

    def batch_predict(self, imgs):
        """推理一个batch的图片.

        Parameters
        ----------
        imgs : list
            一个列表，里面多个np.ndarray类型图片.

        Returns
        -------
        list
            二维列表，每张图片一个列表，包含0到多个BB.

        """
        res_batch = self.model.batch_predict(imgs, transforms=self.transform)
        res = []
        for idx, bbs in enumerate(res_batch):
            res.append([])
            for bb in bbs:
                if bb["score"] > self.thresh:
                    res[-1].append(BB(bb["bbox"], type="pdx"))
                res[-1].sort(key=lambda bb: (bb.wmin, bb.hmin))
        return res

    def add(self, img, info):
        """添加一个推理任务.
        一般需要从视频流中读出多张图像组batch，add和flush操作在推理类内部管理数据，降低代码耦合
        设置bs后当add了bs次后会自动flush，否则返回的结果是三个空list


        Parameters
        ----------
        img : np.ndarray
            一张需要推理的图片.
        info : 任意类型
            需要跟着这张图片的任何信息，bb，下标。。。.

        Returns
        -------
        (list, list, list)
            (所有推理输入图片的list， 所有图片info的list， 所有推理结果的list).

        """
        self.imgs.append(img)
        self.infos.append(info)
        if self.autoflush and len(self.imgs) == self.bs:
            return self.flush()
        else:
            return [], [], []

    def flush(self):
        """推理当前推理对象中所有储存的数据.

        Returns
        -------
        (list, list, list)
            (所有推理输入图片的list， 所有图片info的list， 所有推理结果的list).

        """
        if len(self.imgs) == 0:
            return [], [], []
        res = self.batch_predict(self.imgs)
        imgs = self.imgs.copy()
        self.imgs = []
        infos = self.infos.copy()
        self.infos = []
        return imgs, infos, res

    def __len__(self):
        return len(self.imgs)
