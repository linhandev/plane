"""平均平滑bb四个坐标"""
import os
import os.path as osp
from xml.dom import minidom

import numpy as np

tags = ["xmin", "ymin", "xmax", "ymax"]
ann_path = "/home/lin/Desktop/data/plane/flg/ann/train"
out_path = "/home/lin/Desktop/data/plane/flg/ann/smooth"
dist_thresh = 3
beta = 0.9
def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

files = os.listdir(ann_path)
names = [n.split("_")[0] for n in files]
names = list(set(names))
names.sort()
print(names)
print(len(names))

for name in names:
    print(name)
    fs = [n for n in files if n.startswith(name)]
    fs.sort()
    pos = np.zeros([1,4])
    center_prev = [0,0]
    for t, fname in enumerate(fs):
        xmldoc = minidom.parse(osp.join(ann_path, fname))
        boxes = xmldoc.getElementsByTagName("bndbox")
        box = boxes[0]
        p=[float(box.getElementsByTagName(t)[0].firstChild.data) for t in tags]
        pos_now = np.array(p).reshape([1,4])
        center = [(p[0] + p[2])/2, (p[1] + p[3])/2]
        if dist(center, center_prev) > dist_thresh:
            pos = pos_now
            print("restart at", fname)
        center_prev = center
        pos = (pos * beta + pos_now * (1-beta))
        print(pos_now, pos)

        for idx in range(4):
            xmldoc.getElementsByTagName(tags[idx])[0].childNodes[0].nodeValue = pos[0][idx]
        with open(osp.join(out_path, fname), "w") as f:
            f.write(xmldoc.toxml())
        # input("here")

        # print(pos)

    input("here")






self.bar_value_cache = [new_h_bar_value, new_v_bar_value]

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)
