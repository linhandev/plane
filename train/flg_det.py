import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddle
import paddlex as pdx
from paddlex.det import transforms as t

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

train_transforms = t.Compose([
        t.RandomHorizontalFlip(),
        t.RandomExpand(),
        t.RandomDistort(),
        # t.MixupImage(mixup_epoch=int(epoch_num * 0.5)),
        t.Resize(target_size=width, interp='RANDOM'),
        t.Normalize(mean=mean, std=std),
        ])

train_dataset = pdx.datasets.CocoDetection(
    data_dir='/home/aistudio/data/data67498/DatasetId_153862_1611403574/Images',
    ann_file='/home/aistudio/data/data67498/DatasetId_153862_1611403574/Annotations/coco_info.json',
    transforms=train_transforms,
    num_workers=8,
    buffer_size=256,
    parallel_method='process',
    shuffle=True)
# eval_dataset = pdx.datasets.CocoDetection(
#     data_dir='/home/aistudio/data/data67498/DatasetId_152881_1610856374/Images',
#     ann_file='/home/aistudio/data/data67498/DatasetId_152881_1610856374/val.json',
#     transforms=eval_transforms)

num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='MobileNetV3_large')

model.train(
    num_epochs=100,
    save_interval_epochs=10,
    # pretrain_weights="IMAGENET",
    train_dataset=train_dataset,
    train_batch_size=32,
    # eval_dataset=eval_dataset,
    learning_rate=0.001,
    lr_decay_epochs=[80, 90],
    save_dir='../model/ckpt/flg_det',
    use_vdl=True)
