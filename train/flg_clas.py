import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.cls import transforms
import paddlex as pdx

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotate(),
    transforms.RandomDistort(),
    transforms.Normalize()
])

# eval_transforms = transforms.Compose([
#     transforms.Normalize()
# ])

train_dataset = pdx.datasets.ImageNet(
    data_dir='/home/aistudio/data/data67498/train',
    file_list='/home/aistudio/data/data67498/train/train_list.txt',
    label_list='/home/aistudio/data/data67498/train/labels.txt',
    transforms=train_transforms,
    shuffle=True)

# eval_dataset = pdx.datasets.ImageNet(
#     data_dir='vegetables_cls',
#     file_list='vegetables_cls/val_list.txt',
#     label_list='vegetables_cls/labels.txt',
#     transforms=eval_transforms)

model = pdx.cls.ResNet50_vd_ssld(num_classes=len(train_dataset.labels))

model.train(
    num_epochs=100,
    train_dataset=train_dataset,
    train_batch_size=32,
    # pretrain_weights='IMAGENET',
    # eval_dataset=eval_dataset,
    lr_decay_epochs=[80,90],
    learning_rate=0.0001,
    save_dir='../model/ckpt/flg_clas',
    use_vdl=True)
