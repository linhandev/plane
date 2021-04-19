import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddle
import paddlex as pdx
from paddlex.det import transforms as t

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

train_transforms = t.Compose([
        t.Resize(target_size=512, interp='RANDOM'),
        t.RandomHorizontalFlip(),
        #t.RandomExpand(),
        #t.RandomDistort(),
        #t.MixupImage(mixup_epoch=int(epoch_num * 0.5)),
        t.Normalize(mean=mean, std=std),
        ])

eval_transforms = t.Compose([
        t.Resize(target_size=512, interp='RANDOM'),
        t.RandomHorizontalFlip(),
        #t.RandomExpand(),
        #t.RandomDistort(),
        #t.MixupImage(mixup_epoch=int(epoch_num * 0.5)),
        t.Normalize(mean=mean, std=std),
        ])

train_dataset = pdx.datasets.VOCDetection(
        data_dir='/home/aistudio/data/data81569/flg-new',
        file_list='/home/aistudio/data/data81569/flg-new/train_list.txt',
        label_list='/home/aistudio/data/data81569/flg-new/labels.txt',
        transforms=train_transforms,
        num_workers=4,
        buffer_size=200,
        shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
        data_dir='/home/aistudio/data/data81569/flg-new',
        file_list='/home/aistudio/data/data81569/flg-new/val_list.txt',
        label_list='/home/aistudio/data/data81569/flg-new/labels.txt',
        transforms=eval_transforms,
        num_workers=4,
        buffer_size=200,
        shuffle=True)

num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='MobileNetV3_large')

model.train(
    num_epochs=100,
    save_interval_epochs=10,
    pretrain_weights="IMAGENET",
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    learning_rate=0.001,
    lr_decay_epochs=[80, 90],
    save_dir='../model/ckpt/flg_det',
    use_vdl=True)
