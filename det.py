# def object_detection(paths=None,
#                      images=None,
#                      batch_size=1,
#                      use_gpu=False,
#                      output_dir='detection_result',
#                      score_thresh=0.5,
#                      visualization=True)

import paddlehub as hub
import cv2
import os

imgs = os.listdir("./frames")
imgs = [os.path.join("frames", n) for n in imgs]
print(imgs)

object_detector = hub.Module(name="yolov3_resnet50_vd_coco2017")
for img in imgs:
    result = object_detector.object_detection(
        paths=[img],
        use_gpu=True,
        output_dir="./result",
        visualization=True,
    )
    print(result)
