import random
import os

names = os.listdir("/home/lin/Desktop/data/plane/video/all")
choices = random.sample(names, int(len(names)*0.5))


for c in choices:
    print(c, end=" ")
print("\n")
