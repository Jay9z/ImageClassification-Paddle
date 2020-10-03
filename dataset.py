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

DATASET = "cat_12"

def loader(path):
    x  = Image.open(path).convert("RGB")
    x = np.asarray(x).astype('uint8')
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    #x = cv2.resize(x,(224,224))
    return x

def test_loader(path):
    x  = Image.open(path).convert("RGB")
    x = np.asarray(x).astype('uint8')
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
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
    A.Resize(224,224),
])

test_transform = A.Compose([
    A.Resize(224,224),
])

train_transform = A.Compose([
    A.Resize(224,224),
    A.HueSaturationValue(),
    A.RandomRotate90(),
    A.ShiftScaleRotate(rotate_limit=15),
    A.Blur(),
    A.OpticalDistortion(),
    #A.GridDistortion(),
    A.Cutout(num_holes=16,max_h_size=16,max_w_size=16),
])


## Data file list
train_list = pd.read_csv(f"{DATASET}/train_list.txt",index_col=False,sep='\t').to_numpy()
size = train_list.shape[0]
ratio = 0.8
offset = int(size*ratio)
np.random.shuffle(train_list)
logger.info("after shuffle: {}".format(train_list[:,1]))

test_list_temp = glob(f"{DATASET}/test/*.jpg|png")
test_list = []
for path in test_list_temp:
    test_list.append(path[len(DATASET)+1:])

## batch/shuffle dataset with Reader
def train_reader():
    _data = train_list[:offset]
    for i in range(len(_data)):
        img, label = loader(f"{DATASET}/"+ _data[i,0]),_data[i,1]
        img = train_transform(image=img)['image']
        #print(np.max(img),np.min(img))
        #img = transform_ops(img)[0]
        img = img.transpose(2,0,1)/255.0
        yield img, label

def val_reader():
    _data = train_list[offset:]
    for i in range(len(_data)):
        img, label = test_loader(f"{DATASET}/"+ _data[i,0]),_data[i,1]
        img = val_transform(image=img)['image']/255.0
        img = img.transpose(2,0,1)
        yield img, label

def test_reader():
    _data = test_list
    for i in range(len(_data)):
        img = test_loader(f"{DATASET}/"+ _data[i])
        img = test_transform(image=img)['image']
        img = img.transpose(2,0,1)/255.0
        yield img

if __name__ == "__main__":
    for i,x_y in enumerate(train_reader()):
        logger.info("batch reader {}".format(x_y[0][0].shape))
        image,label = x_y
        image = np.array(image).transpose((1,2,0))
        show_image(image,label)
        cv2.imwrite(f"train_{i}.jpg",image*255.0)
        if i == 8:
            break;

    for x_y in val_reader():
        logger.info("val_data {}".format(x_y[0][0].shape))
        image,label = x_y
        image = np.array(image).transpose((1,2,0))
        show_image(image,label)
        cv2.imwrite("val.jpg",image*255.0)
        break;

    for x_y in test_reader():
        logger.info("test data shape: {}".format( x_y.shape))
        logger.info("{}".format(x_y))
        image = x_y
        show_image(image," ")
        break;