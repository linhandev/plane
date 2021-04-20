from math import hypot
from xml.dom import minidom

import cv2
import numpy as np
import pandas as pd


def toint(l):
    return [int(x) for x in l]


class Stream:
    def __init__(self, vid_path, toi_path=None, itv_sparse=25, itv_dense=3, start_frame=None):
        """创建视频流.
        Parameters
        ----------
        vid_path : str
            视频流地址，cv2.VideoCapture能接受的任何流都行.
        toi_path : str
            如果有视频感兴趣时间区域的文件，写路径.
        itv_sparse : int
            稀疏抽帧间隔，在toi外面按这个间隔抽帧；如果为0则略过所有toi外的帧.
        itv_dense : int
            密集抽帧间隔，在toi里边按照这个间隔抽帧；如果为0略过所有toi内的帧.
        start_frame : int
            从第几帧开始，调试功能，这个最后设置idx，会override toi_only.
        Returns
        -------
        type
            一个可以下标索引，可以迭代的视频流对象.
        """

        # TODO: 处理异常,判断打开成功
        vid = cv2.VideoCapture(vid_path)
        if start_frame is not None:
            self.idx = start_frame
        else:
            self.idx = 0
        self.sitv = itv_sparse
        self.ditv = itv_dense
        self.type = None
        self.fps = int(vid.get(cv2.CAP_PROP_FPS))
        self.shape = [  # HWC
            vid.get(cv2.CAP_PROP_FRAME_HEIGHT),
            vid.get(cv2.CAP_PROP_FRAME_WIDTH),
            3,
        ]
        self.shape = toint(self.shape)
        self.frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        # if vid_path.find("/") == -1:
        vid_name = vid_path.split("\\")[-1]
        # else:
        #     vid_name = vid_path.split("/")[-1]
        # 获取 toi
        # TODO: 修改成读取json格式
        if toi_path is not None:
            time_mark = pd.read_csv(toi_path)
            if vid_name in time_mark["name"].values:
                df = time_mark[time_mark["name"].isin(vid_name.split())].iloc[:, 1:]
                info = df.values.tolist()[0]
                info.insert(0, 0)
                self.toi = [x * self.fps for x in info]
                self.toi.append(self.frame_count)
            else:
                self.toi = [0, self.frame_count]
        else:
            self.toi = [0, self.frame_count]

        # TODO: 更精准的计算len
        self.len = 0
        if self.sitv != 0:
            for idx in range(0, len(self.toi), 2):
                self.len += (self.toi[idx + 1] - self.toi[idx]) / self.sitv
        if self.ditv != 0:
            for idx in range(1, len(self.toi) - 1, 2):
                self.len += (self.toi[idx + 1] - self.toi[idx]) / self.ditv
        self.len = int(self.len)
        self.vid = vid

    def __getitem__(self, idx):
        """使其支持[].
        按照frame取帧
        Parameters
        ----------
        idx : int
            要第几帧的数据.
        Returns
        -------
        bool, np.ndarray
            是否获取成功，第 idx 帧的数据.
        """
        # if idx >= self.frame_count:
        #     raise KeyError("Index {} exceed frame count".format(idx))
        idx = min(self.frame_count - 1, idx)
        self.vid.set(1, idx)
        return self.vid.read()

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def in_toi(self):
        """判断当前帧是否在toi里.
        Returns
        -------
        bool, int
            bool是当前在不在toi里，int是这个区间(不一定在不在toi里)结束的时间.
        """
        # print(self.idx, self.frame_count)
        for idx in range(len(self.toi) - 1):
            if self.toi[idx] <= self.idx < self.toi[idx + 1]:
                return not (idx % 2 == 0), self.toi[idx + 1]

        raise Exception("toi不应该判断最后一帧")

    def __next__(self):
        """迭代支持.
        Returnstype
        -------
        tuple
            当前帧的id和帧图片.
        """
        curr_idx = self.idx
        # BUG: 有的视频会在序列最后出现最后一帧
        if self.idx >= self.frame_count:
            raise StopIteration
        is_in, next_idx = self.in_toi()
        if is_in:
            if self.ditv != 0:
                self.idx += self.ditv
                # print("In toi, curr idx: {}".format(self.idx))
            else:
                self.idx = next_idx
        else:
            if self.sitv != 0:
                self.idx += self.sitv
            else:
                self.idx = next_idx
        success, img = self[self.idx]
        if not success:
            raise StopIteration
        return curr_idx, img, is_in


class BB:
    # TODO: 用getter setter实现 wr，hr，返回一个slice
    """
    x是宽,y是高
    cv2里numpy下标是 HWC
    有选择的时候一律写 WHC
    """

    wmin = 0
    wmax = 0
    hmin = 0
    hmax = 0

    def __init__(self, p, type="WH", size=(None, None)):
        """创建一个bb.
        Parameters
        ----------
        p : list
            四个位置.
        type : str
            不同的格式.
            pdx: w左上角,h左上角,w长度,h长度
        size : tuple
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
        self.list = [self.wmin, self.hmin, self.wmax, self.hmax]
        if size[0] is not None:
            self.size = [int(t) for t in size]
        else:
            self.size = (None, None)

    def __pos__(self):
        """在实例前加+的结果.
        Returns
        -------
        BB
            四个数都大于0的bb.
        """
        return BB(
            [max(0, self.wmin), max(0, self.hmin), max(0, self.wmax), max(0, self.hmax)],
            size=self.size,
        )

    def spill(self):
        if self.wmin < 0 or self.hmin < 0:
            return True
        if self.size != (None, None) and (self.wmax > self.size[0] or self.hmax > self.size[1]):
            return True
        return False

    def __getitem__(self, idx):
        return self.list[idx]

    def __iter__(self):
        return self.list.__iter__()

    def __repr__(self):
        """打印支持.
        Returns
        -------
        type
            Description of returned object.
        """
        return "BB: WHC ({}, {}), ({}, {}) Image size: {}".format(
            self.wmin, self.wmax, self.hmin, self.hmax, self.size
        )

    def __gt__(self, size):
        if (self.wmax - self.wmin) > size[0] or (self.hmax - self.hmin) > size[1]:
            return True
        return False

    def __lt__(self, size):
        if (self.wmax - self.wmin) < size[0] or (self.hmax - self.hmin) < size[1]:
            return True
        return False

    def __neq__(self, size):
        if (self.wmax - self.wmin) != size[0] or (self.hmax - self.hmin) != size[1]:
            return True
        return False

    def square(self, length):
        """返回一个和self同中心，length边长的bb.
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
        wl, hl = self.wc - l + 0.1, self.hc - l + 0.1
        wh, hh = self.wc + l + 0.1, self.hc + l + 0.1
        return BB([wl, hl, wh, hh], "WH", self.size)

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
        return BB([self.wc - wl, self.hc - hl, self.wc + wl, self.hc + hl], "WH", self.size)

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


def xml2bb(path, obj_type="person"):
    tags = ["xmin", "ymin", "xmax", "ymax"]
    xmldoc = minidom.parse(path)
    objs = xmldoc.getElementsByTagName("object")
    size = xmldoc.getElementsByTagName("size")[0]
    width = size.getElementsByTagName("width")[0].firstChild.data
    height = size.getElementsByTagName("height")[0].firstChild.data

    bbs = []
    for obj in objs:
        type = obj.getElementsByTagName("name")[0].firstChild.data
        if type == obj_type:
            p = [float(obj.getElementsByTagName(t)[0].firstChild.data) for t in tags]
            bbs.append(BB(p, "WH", (width, height)))
    return bbs


def crop(img, b, do_pad=False):
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
    pad = [0, 0, 0, 0]
    b = list(b)
    # TODO: 测试pad的代码对不对
    for idx in range(2):
        if b[idx] < 0:
            pad[idx] = -b[idx]
            b[idx] = 0

    shape = img.shape[1::-1]
    for idx in range(2, 4):
        if b[idx] > shape[idx - 2]:
            pad[idx] = b[idx] - shape[idx - 2]
            b[idx] = shape[idx - 2]
    ret = img[b[1] : b[3], b[0] : b[2], :]
    if do_pad:
        ret = np.pad(ret, [[pad[1], pad[3]], [pad[0], pad[2]], [0, 0]], "reflect")
    return ret


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
        cv2.line(img, l[0], l[1], color, 1)


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


def get_xml(pos, type="flg"):
    res = f"""
    <annotation>
        <folder>image</folder>
        <filename></filename>
        <path></path>
        <source>
                <database></database>
        </source>
        <size>
                <width>1920</width>
                <height>1080</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
                <name>{type}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>{pos[0]}</xmin>
                        <ymin>{pos[1]}</ymin>
                        <xmax>{pos[2]}</xmax>
                        <ymax>{pos[3]}</ymax>
                </bndbox>
        </object>
    </annotation>
    """
    return res