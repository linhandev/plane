from paddle.nn import Conv2D, BatchNorm2D, ReLU, Softmax, MaxPool2D, Flatten, Linear


ClasModel = paddle.nn.Sequential(
    Conv2D(3, 6, (3, 3)),
    BatchNorm2D(6),
    ReLU(),
    Conv2D(6, 6, (3, 3)),
    BatchNorm2D(6),
    ReLU(),
    MaxPool2D((2, 2)),
    Conv2D(6, 12, (3, 3)),
    BatchNorm2D(12),
    ReLU(),
    Conv2D(12, 12, (3, 3)),
    BatchNorm2D(12),
    ReLU(),
    MaxPool2D((2, 2)),
    Conv2D(12, 8, (3, 3)),
    BatchNorm2D(8),
    ReLU(),
    Conv2D(8, 8, (3, 3)),
    BatchNorm2D(8),
    ReLU(),
    MaxPool2D((2, 2)),
    Flatten(),
    Linear(128, 128),
    ReLU(),
    Linear(128, 32),
    ReLU(),
    Linear(32, 2),
    Softmax(),
)

model = paddle.Model(ClasModel)
model.load("../model/best/person_clas/person_clas")

img = cv2.imread("/home/aistudio/plane/bend/p/15351-撤轮挡-0_2937.png")
print(model.predict(img))
