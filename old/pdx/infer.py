import paddlex as pdx
from paddlex.det import transforms

model = pdx.load_model('output/yolov3/epoch_40')
image_name = '/home/aistudio/data/data67498/DatasetId_152881_1610856374/Images/1.png'
transforms = transforms.Compose([
    transforms.Resize(), transforms.Normalize()
])

# transforms = transforms.Compose([transforms.ComposedYOLOv3Transforms()])


# result = model.predict(image_name, transforms=transforms)

result = model.batch_predict([], transforms=transforms)


print(result)
pdx.det.visualize(image_name, result, threshold=0.001, save_dir='./output/plane_lg')


'''
Transforms:
- Resize:
    interp: CUBIC
    target_size: 608
- Normalize:
    mean:
    - 0.485
    - 0.456
    - 0.406
    std:
    - 0.229
    - 0.224
    - 0.225

'''