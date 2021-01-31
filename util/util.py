from math import hypot

import cv2


class Stream:
    def __init__(self, vid_path, toi_path=None, itv_sparse=25, itv_dense=5, start_frame=0):
        """创建视频流.

        Parameters
        ----------
        vid_path : str
            视频流地址，cv2.VideoCapture能接受的任何流都行.
        toi_path : str
            如果有视频感兴趣时间区域的文件，写路径.
        itv_sparse : int
            稀疏抽帧间隔，在toi外面按这个间隔抽帧.
        itv_dense : int
            密集抽帧间隔，在toi里边按照这个间隔抽帧.
        start_frame : int
            从第几帧开始，基本是调试功能.

        Returns
        -------
        type
            一个可以下标索引，可以迭代的视频流对象.
        """

        # TODO: 处理异常,判断打开成功
        vid = cv2.VideoCapture(vid_path)
        self.idx = start_frame
        self.sitv = itv_sparse
        self.ditv = itv_dense
        self.fstart = self.fend = 0
        self.type = None
        self.fps = vid.get(cv2.CAP_PROP_FPS)
        self.size = [
            vid.get(cv2.CAP_PROP_FRAME_WIDTH),
            vid.get(cv2.CAP_PROP_FRAME_HEIGHT),
        ]
        self.frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) / self.sitv) - 1

        if toi_path is not None:
            with open(toi_path, "r") as f:
                time_str = f.read()
            self.fstart, self.fend, self.type = time_str.split(" ")
            self.fstart = int(self.fstart)
            self.fend = int(self.fend)
            self.fstart *= self.fps
            self.fend *= self.fps

        self.vid = vid

    def __getitem__(self, idx):
        """使其支持[].
        按照稀疏间隔取帧

        Parameters
        ----------
        idx : int
            要第几个sitv的数据.

        Returns
        -------
        type
            返回 idx × sitv 帧的数据.

        """
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
        """迭代支持.

        Returnstype
        -------
        tuple
            当前帧的id和帧图片.

        """
        # TODO: 添加vidcapture支持
        if self.fstart <= self.idx <= self.fend:
            self.idx += self.ditv
            print("In toi, curr idx: {}".format(self.idx))
        else:
            self.idx += self.sitv
        self.vid.set(1, self.idx)
        success, image = self.vid.read()
        if not success:
            raise StopIteration
        return self.idx, image


class BB:
    """
    x是宽,y是高
    cv2里numpy下标是 HWC
    有选择的时候一律写 WHC
    """

    wmin = 0
    wmax = 0
    hmin = 0
    hmax = 0

    def __init__(self, p, type="WH", size=[None, None]):
        """创建一个bb.

        Parameters
        ----------
        p : list
            四个位置.
        type : str
            不同的格式.
            pdx: w左上角,h左上角,w长度,h长度
        size : tuple

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
        """打印支持.

        Returns
        -------
        type
            Description of returned object.

        """
        return "BB: WHC ({}, {}), ({}, {})".format(self.wmin, self.wmax, self.hmin, self.hmax)

    def square(self, length, restrict=False):
        """返回一个和self同中心，length边长的bb.
        如果有宽度和高度，可以限制bb不出图片

        Parameters
        ----------
        length : int
            目标正方形bb的边长.

        Returns
        -------
        BB
            和self同中心，length边长的bb.

        """
        l = length // 2
        if restrict and self.width is not None and self.height is not None:
            wl, hl = max(self.wc - l, 0), max(self.hc - l, 0)
            wh, hh = min(self.wc + l, self.width), min(self.hc + l, self.height)
        else:
            wl, hl = self.wc - l, self.hc - l
            wh, hh = self.wc + l, self.hc + l
        return BB([wl, hl, wh, hh], "WH")

    def region(self, ratio):
        """返回一个和self同中心，宽，长分别为ratio倍的bb.

        Parameters
        ----------
        ratio : list/tuple
            目标bb宽和长分别为当前bb的多少倍.

        Returns
        -------
        BB
            和self同中心，宽，长分别为ratio倍的bb.

        """
        wl = self.wmax - self.wmin
        hl = self.hmax - self.hmin
        r = ratio
        wl = int(wl * r[0] / 2)
        hl = int(hl * r[1] / 2)
        return BB([self.wc - wl, self.hc - hl, self.wc + wl, self.hc + hl], "WH")

    def center(self):
        return [self.wc, self.hc]

    def contains(self, obj):
        """当前bb是否包含一个点或者另一个bb中心.

        Parameters
        ----------
        obj : BB/list/tuple
            一个BB或者一个点.

        Returns
        -------
        Bool
            当前bb是否包含一个点或者另一个bb中心.

        """
        if isinstance(obj, BB):
            p = obj.center()
        elif isinstance(obj, list):
            p = obj
        if self.wmin <= p[0] <= self.wmax and self.hmin <= p[1] <= self.hmax:
            return True
        return False

    def transpose(self, bb):
        """在当前box里面做检测，将得到的结果bb转换为当前bb所在坐标系的坐标.
        比如当前bb是起落架周围256区域，在这里面检测到一个person_bb，则 self.transpose(person_bb) 得到这个人bb在整张图上的
        位置

        Parameters
        ----------
        bb : BB
            self内部的一个bb.

        Returns
        -------
        type
            变换成大坐标系bb的位置.

        """
        return BB(
            [
                self.wmin + bb.wmin,
                self.hmin + bb.hmin,
                self.wmin + bb.wmax,
                self.hmin + bb.hmax,
            ],
            "WH",
        )


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
    # TODO: 对出界的部分进行pad
    return img[b.hmin : b.hmax, b.wmin : b.wmax, :]


def dpoint(img, p, color="R"):
    """在img的p位置上画一个color颜色的点.

    Parameters
    ----------
    img : np.ndarray
        图片.
    p : list
        按照cv2格式，WH.
    color : str
        R,G,B.

    """
    if color == "R":
        color = (0, 0, 255)
    elif color == "G":
        color = (0, 255, 0)
    elif color == "B":
        color = (255, 0, 0)
    cv2.circle(img, (p[0], p[1]), 1, color, 4)


def dbb(img, b, color="R"):
    """在img上b范围画一个color颜色的bounding box.

    Parameters
    ----------
    img : np.ndarray
        图片.
    b : BB
        BB instance.
    color : str
        R,G,B.

    """
    xmin, ymin, xmax, ymax = b.wmin, b.hmin, b.wmax, b.hmax
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
    """在图像左上角画红/绿色块，表示分类结果.

    Parameters
    ----------
    img : np.ndarray
        需要标识的图像.
    res : Bool
        分类的结果.

    """
    if img.shape[0] <= 64:
        img = img[8:16, 8:16, :]
    else:
        img = img[32:64, 32:64, :]
    img = 0
    if res:
        img[:, :, 1] = 255
    else:
        img[:, :, 2] = 255
