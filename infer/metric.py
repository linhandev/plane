import os
import os.path as osp

import numpy as np

res_dir = "/home/aistudio/plane/帧推理结果/"
ann_dir = "/home/aistudio/plane/时间标注/all/"
tp_all = 0
fp_all = 0
for res_name in os.listdir(res_dir):
    # vid_name = res_name[:-4] + ".mp4"
    # print(vid_name, res_name)
    ann_path = osp.join(ann_dir, res_name)
    with open(ann_path, "r") as f:
        ann = f.read()
        if len(ann) == 0:
            continue
    ann = ann.split(" ")
    ann = [int(ann[0]) * 25, int(ann[1]) * 25]
    # print(ann)
    res_path = osp.join(res_dir, res_name)

    with open(res_path, "r") as f:
        res = f.readlines()
        res = [d.split(" ") for d in res]
        res = [[int(d[0]), 1 if d[1] == "True" else 0] for d in res]
        # print(res)

    window = 8
    thresh = 7 / 8
    mem = [0 for _ in range(window)]
    ap = 0
    tp = 0
    for idx, predict in res:
        mem[0] = predict
        for i in range(window - 1, 0, -1):
            mem[i] = mem[i - 1]
        tot = np.sum(mem)
        if tot >= window * thresh:
            # print(idx, tot)
            ap += 1
            if ann[0] < idx < ann[1]:
                tp += 1
    # print(ap, tp, ap - tp)
    if ap == 0:
        tpr = 0
        fpr = 0
    else:
        tpr = tp / ap
        fpr = (ap - tp) / ap
    print(f"{res_name}\t{tpr}\t{fpr}\t{tpr+fpr}")
    tp_all += tpr
    fp_all += fpr


print(tp_all / len(os.listdir(res_dir)))
print(fp_all / len(os.listdir(res_dir)))
