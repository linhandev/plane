# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms as t
import paddlex as pdx
import paddle

# # 下载和解压昆虫检测数据集
# insect_dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
# pdx.utils.download_and_decompress(insect_dataset, path='./')

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html

# train_transforms = transforms.Compose([
#     transforms.Resize([1920, 1080]), transforms.RandomDistort(), transforms.RandomHorizontalFlip(), transforms.Normalize()
# ])

# eval_transforms = transforms.Compose([
#     transforms.Resize([1920, 1080]), transforms.Normalize()
# ])


# train_transforms = t.Compose([t.ComposedYOLOv3Transforms("train")])
# eval_transforms = t.Compose([t.ComposedYOLOv3Transforms("eval")])

width = 480
height = 270
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
epoch_num = 15

train_transforms = t.Compose([
        t.RandomHorizontalFlip(),
        t.RandomExpand(),
        t.RandomDistort(),
        t.MixupImage(mixup_epoch=int(epoch_num * 0.5)),
        t.Resize(target_size=width, interp='RANDOM'), 
        t.Normalize(mean=mean, std=std),
        ])
# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.CocoDetection(
    data_dir='/home/aistudio/data/data67498/DatasetId_153212_1611125306/Images',
    ann_file='/home/aistudio/data/data67498/DatasetId_153212_1611125306/Annotations/coco_info.json',
    transforms=train_transforms,
    num_workers=4,
    buffer_size=64,
    parallel_method='process',
    shuffle=True)
# eval_dataset = pdx.datasets.CocoDetection(
#     data_dir='/home/aistudio/data/data67498/DatasetId_152881_1610856374/Images',
#     ann_file='/home/aistudio/data/data67498/DatasetId_152881_1610856374/val.json',
#     transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
# print(eval_dataset)
num_classes = len(train_dataset.labels)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3
# model = pdx.det.PPYOLO(num_classes=num_classes)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='MobileNetV3_large')

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=epoch_num,
    save_interval_epochs=1,
    train_dataset=train_dataset,
    train_batch_size=32,
    # optimizer= paddle.fluid.optimizer.AdamOptimizer,
    # eval_dataset=eval_dataset,
    learning_rate=0.001,
    lr_decay_epochs=[20, 170, 180, 190],
    save_dir='output/yolov3',
    use_vdl=True)
