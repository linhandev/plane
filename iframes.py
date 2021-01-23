import os
import os.path as osp

import paddlex as pdx
from paddlex.det import transforms


vid_dir = "/home/aistudio/plane/vid-split/train"
time_dir = "/home/aistudio/plane/time/all"
bs = 16
itv = 3


flg_det = pdx.load_model("   ")
transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])


for vid_name in os.listdir(vid_dir):
    print(vid_name)

    cv2.VideoCapture(osp.join(vid_dir, vid_name))
    vid_name = vid_name.split(".")[0]
    with open(osp.join(time_dir, vid_name + ".txt"), "r") as f:
        times = f.read()
    s, e, action = times.split(" ")
    print(s, e, action)

    img_data = []
    index = []
    for idx in range(int(s)*25, int(e)*25, itv):
        vidcap.set(1, idx)
        success, image = vidcap.read()
        if success:
            img_data.append(image)
            index.append(idx)
            if len(img_data) == bs or idx + itv >= int(e)*25:
                flgs = flg_det.batch_predict(img_data, transforms=transforms)
                for flg in flgs:
                    print(flg)

                # vid_name + str(idx).zfill(6) + '.png')
