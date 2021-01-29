import cv2
import paddlex as pdx
from paddlex.det import transforms as dT


class PdxDet:
    imgs = []
    names = []

    def __init__(self, model_path="../model/best/flg_det/", bs=2):
        self.model = pdx.load_model(model_path)
        self.transform = dT.Compose([dT.Resize(), dT.Normalize()])
        self.bs = bs

    def predict(self, img):
        bbs = self.model.predict(img, transforms=self.transform)
        res = []
        if len(bbs) != 0:
            for bb in bbs:
                res.append(BB(bb["bbox"], type="pdx"))
        return res

    def batch_predict(self, imgs):
        res_batch = self.model.batch_predict(imgs, transforms=self.transform)
        res = []
        for idx, bbs in enumerate(res_batch):
            res.append([])
            for bb in bbs:
                res[-1].append(BB(bb["bbox"], type="pdx"))
        return res

    def add(self, img, name):
        self.imgs.append(img)
        self.names.append(str(name))
        if len(self.imgs) == self.bs:
            return self.flush()
        else:
            return [], [], []

    def flush(self):
        res = self.batch_predict(self.imgs)
        imgs = self.imgs.copy()
        self.imgs = []
        names = self.names.copy()
        self.names = []
        return imgs, names, res


class Stream:
    def __init__(self, path, itv=25, start_frame=0):
        # TODO: 处理异常,判断打开成功
        vid = cv2.VideoCapture(path)
        self.idx = start_frame
        self.itv = itv
        self.fps = vid.get(cv2.CAP_PROP_FPS)
        self.size = [
            vid.get(cv2.CAP_PROP_FRAME_WIDTH),
            vid.get(cv2.CAP_PROP_FRAME_HEIGHT),
        ]
        self.frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) / self.itv) - 1
        self.vid = vid

    def __getitem__(self, idx):
        # TODO: 研究为什么只能到 -2itv
        idx = min(self.frame_count - self.itv * 2, idx * self.itv)
        self.vid.set(1, idx)
        success, image = self.vid.read()
        return image

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += self.itv
        self.vid.set(1, self.idx)
        success, image = self.vid.read()
        if not success:
            raise StopIteration
        return image


class BB:
    """
    x是宽,y是高
    cv2里numpy下标是 HWC
    """

    wmin = 0
    wmax = 0
    hmin = 0
    hmax = 0

    def __init__(self, p, type="WH"):
        """创建一个bb.

        Parameters
        ----------
        p : list
            四个位置.
        type : str
            不同的格式.
            pdx: w左上角,h左上角,w长度,h长度

        Returns
        -------
        type
            Description of returned object.

        """
        p = [int(t) for t in p]
        if type == "pdx":
            self.wmin = int(p[0])
            self.hmin = int(p[1])
            self.wmax = int(p[0] + p[2])
            self.hmax = int(p[1] + p[3])
        elif type == "WH":
            self.wmin = int(p[0])
            self.hmin = int(p[1])
            self.wmax = int(p[2])
            self.hmax = int(p[3])
        elif type == "HW":
            self.hmin = int(p[0])
            self.wmin = int(p[1])
            self.hmax = int(p[2])
            self.wmax = int(p[3])
        else:
            pass

        self.wc = (self.wmin + self.wmax) // 2
        self.hc = (self.hmin + self.hmax) // 2

    def __repr__(self):
        return "BB: WHC ({}, {}), ({}, {})".format(
            self.wmin, self.wmax, self.hmin, self.hmax
        )

    def square(self, length):
        l = length // 2
        return BB([self.wc - l, self.hc - l, self.wc + l, self.hc + l], "WH")

    def region(self, ratio):
        wl = self.wmax - self.wmin
        hl = self.hmax - self.hmin
        r = ratio
        wl = int(wl * r[0] / 2)
        hl = int(hl * r[1] / 2)
        return BB([self.wc - wl, self.hc - hl, self.wc + wl, self.hc + hl], "WH")


def crop(img, b):
    """从图像中切下bb范围.

    Parameters
    ----------
    img : np.ndarray
        HWC.
    b : BB
        Description of parameter `bb`.

    Returns
    -------
    type
        切下来的部分.
    """
    return img[b.hmin : b.hmax, b.wmin : b.wmax, :]


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


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
    cv2.circle(img, (p[1], p[0]), 1, color, 4)


def dbb(img, b, color="R"):
    ymin, xmin, ymax, xmax = b
    lines = [
        [(xmin, ymin), (xmin, ymax)],
        [(xmax, ymin), (xmax, ymax)],
        [(xmin, ymin), (xmax, ymin)],
        [(xmin, ymax), (xmax, ymax)],
    ]
    if color == "R":
        color = (0, 0, 255)
    elif color == "G":
        color = (0, 255, 0)
    elif color == "B":
        color = (255, 0, 0)

    for l in lines:
        cv2.line(img, l[0], l[1], color, 2)


def dpn(img, res):
    if img.shape[0] <= 64:
        img = img[8:16, 8:16, :]
    else:
        img = img[32:64, 32:64, :]
    img = 0
    if res:
        img[:, :, 1] = 255
    else:
        img[:, :, 2] = 255
