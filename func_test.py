#!/usr/bin/env python
# coding: utf-8

import cv2
from PIL import Image
import numpy as np

# ## PIL vs cv2
# #path = r'work/cat_12_train/EdBGUWeiua6mlkC0FJDI4N5wgYjRXZTH.jpg'
# path = r'work/cat_12_train/tO6cKGH8uPEayzmeZJ51Fdr2Tx3fBYSn.jpg'
# print(path)
# #img = cv2.imread(path)
# img  = Image.open(path).convert("RGB")
# img = np.asarray(img).astype('float32')
# print(img)

# ##Transform testing
# path = r'work/cat_12_train/0aSixIFj9X73z41LMDUQ6ZykwnBA5YJW.jpg'
# print(path)
# #img = cv2.imread(path)
# img  = Image.open(path).convert("RGB")
# img = np.asarray(img).astype('float32')
# print(img)

# import os
# import cv2
# from dataset import transform
# if not os.path.exists("aug"):
#     os.makedirs("aug")
# for i in range(1000):
#     print(img.shape)
#     img2 = transform(img)
#     img3 = np.uint8(img2)
#     img3 = cv2.cvtColor(img3,cv2.COLOR_RGB2BGR)
#     cv2.imwrite(f"aug/{i}.jpg",img3)

#     # img3 = Image.fromarray(np.uint8(img2))
#     # img3.save(f"aug/{i}.jpg",quality=95)


##plot test
import numpy as np
from matplotlib import pyplot as plt
x = np.arange(1,11)
y = 2*x + 5

plt.title("matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y)
plt.show