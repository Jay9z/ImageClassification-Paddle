#!/usr/bin/env python
# coding: utf-8
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from paddle.fluid.io import DataLoader
import os
#from paddlex.cls import transforms
import albumentations as A
from utilis import *
from glob import glob

def loader(path):
    x  = Image.open(path).convert("RGB")
    x = np.asarray(x).astype('float32')
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)/255.0
    #x = cv2.resize(x,(224,224))
    return x

def test_loader(path):
    x  = Image.open(path).convert("RGB")
    x = np.asarray(x).astype('float32')
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)/255.0
    #x = cv2.resize(x,(224,224))
    return x

# transform_ops = transforms.Compose(
#     [
#     #transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])   # this will do 1/255.0
#     transforms.RandomHorizontalFlip(prob=0.5),
#     transforms.RandomRotate(rotate_range=30, prob=0.5),
#     transforms.RandomCrop(crop_size=224, lower_scale=0.9, lower_ratio=9. / 10, upper_ratio=10. / 9),
#     transforms.RandomDistort(brightness_range=0.1, brightness_prob=0.5, contrast_range=0.1, contrast_prob=0.5, saturation_range=0.01, saturation_prob=0.5, hue_range=0.01, hue_prob=0.5)
#     ]
# )

val_transform = A.Compose([
    A.Resize(224,224)

])

test_transform = A.Compose([
    A.Resize(224,224),
])

train_transform = A.Compose([
    A.Resize(224,224),
    A.RandomRotate90(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    A.Blur(blur_limit=3),
    A.OpticalDistortion(),
    A.GridDistortion(),
    A.HueSaturationValue(),
    A.Cutout(),
    #A.Transpose(),
])


## Data file list
train_list = pd.read_csv("work/train_list.txt",index_col=False,sep='\t').to_numpy()
size = train_list.shape[0]
ratio = 0.8
offset = int(size*ratio)
np.random.shuffle(train_list)
logger.info("after shuffle: {}".format(train_list[:,1]))

test_list_temp = glob("work/cat_12_test/*.jpg")
test_list = []
for path in test_list_temp:
    test_list.append(path[5:])

## batch/shuffle dataset with Reader
def train_reader():
    _data = train_list[:offset]
    for i in range(len(_data)):
        img, label = loader("work/"+ _data[i,0]),_data[i,1]
        img = train_transform(image=img)['image']
        img = img.transpose(2,0,1)
        yield img, label

def val_reader():
    _data = train_list[offset:]
    for i in range(len(_data)):
        img, label = test_loader("work/"+ _data[i,0]),_data[i,1]
        img = val_transform(image=img)['image']
        img = img.transpose(2,0,1)
        yield img, label

def test_reader():
    _data = test_list
    for i in range(len(_data)):
        img = test_loader("work/"+ _data[i])
        img = test_transform(image=img)['image']
        img = img.transpose(2,0,1)
        yield img

if __name__ == "__main__":
    for x_y in train_reader():
        logger.info("batch reader {}".format(x_y[0][0].shape))
        image,label = x_y[0]
        show_image(image,label)
        break;

    for x_y in val_reader():
        logger.info("val_data {}".format(x_y[0][0].shape))
        image,label = x_y[0]
        show_image(image,label)
        break;

    for x_y in test_reader():
        logger.info("test data shape: {}".format( x_y.shape))
        logger.info("{}".format(x_y))
        image = x_y
        show_image(image," ")
        break;