# from util import xml2bb

# bbs = xml2bb("/home/aistudio/plane/人检测/Annotations/4734.xml")
# for bb in bbs:
#     print(bb)


# import numpy as np
# from util import BB, crop

#
# b = BB([0, 0, 63, 66], size=(64, 64))
# print(list(b))
# img = crop(np.ones([64, 64, 3]), b)
# print(img.shape)


# from util import BB
#
# a = BB([0, 0, 65, 20])
# print(a > (128, 128))


from util import BB

a = BB([0, 0, 64, 64], size=(64, 64))
print(a.spill())
