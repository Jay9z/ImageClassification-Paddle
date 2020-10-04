#!/usr/bin/env python
# coding: utf-8
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# from paddlex.cls import transforms
import albumentations as A
from utilis import show_image, logger
from glob import glob

DATASET = "cat_12"


def loader(path):
    x = Image.open(path).convert("RGB")
    x = np.asarray(x).astype("uint8")
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x


class Dataset():
    def __init__(self):
        
        self.val_transform = A.Compose(
            [
                A.Resize(224, 224),
            ]
            )

        self.test_transform = A.Compose(
            [
                A.Resize(224, 224),
            ]
            )

        self.train_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.HueSaturationValue(),
                A.RandomRotate90(),
                A.ShiftScaleRotate(rotate_limit=15),
                A.Blur(),
                A.OpticalDistortion(),
                # A.GridDistortion(),
                A.Cutout(num_holes=16, max_h_size=16, max_w_size=16),
            ]
            )

        # split ratio between train and val dataset
        ratio = 0.8

        # Data file list
        self.train_list = pd.read_csv(
            f"{DATASET}/train_list.txt", index_col=False, sep="\t"
                ).to_numpy()
        size = self.train_list.shape[0]
        self.offset = int(size * ratio)
        np.random.shuffle(self.train_list)
        logger.info("after shuffle: {}".format(self.train_list[:, 1]))

        test_list_temp = glob(f"{DATASET}/test/*.jpg")
        self.test_list = []
        for path in test_list_temp:
            self.test_list.append(path[len(DATASET) + 1 :])
    
    def get_train_list(self):
        '''
        return: ndarray of 
            [[path1, idx1],
            [path2, idx2],
            ...
            [pathn, idxn]]
        '''
        return self.train_list

    def get_test_list(self):
        '''
        return: ndarray of 
            [path1,path2,...,pathn]
        '''
        return self.test_list

    def get_class_number(self):
        print(self.train_list[:,1])
        indexs = self.train_list[:,1]#.astype("uint8")
        return int(np.max(indexs)-np.min(indexs)+1)

    def train_reader(self):
        _list = self.train_list[:self.offset]
        for i in range(len(_list)):
            img, label = loader(f"{DATASET}/" + _list[i, 0]), _list[i, 1]
            img = self.train_transform(image=img)["image"]
            img = img.transpose(2, 0, 1) / 255.0
            yield img, label


    def val_reader(self):
        _list = self.train_list[self.offset:]
        for i in range(len(_list)):
            img, label = loader(f"{DATASET}/" + _list[i, 0]), _list[i, 1]
            img = self.val_transform(image=img)["image"] / 255.0
            img = img.transpose(2, 0, 1)
            yield img, label


    def test_reader(self):
        _list = self.test_list
        for i in range(len(_list)):
            img = loader(f"{DATASET}/" + _list[i])
            img = self.test_transform(image=img)["image"]
            img = img.transpose(2, 0, 1) / 255.0
            yield img, _list[i]


if __name__ == "__main__":
    dataset = Dataset()
    print(dataset.get_class_number())
    for i, x_y in enumerate(dataset.train_reader()):
        image, label = x_y
        print("train image {}".format(image.shape))
        image = np.array(image).transpose((1, 2, 0))
        show_image(image, label)
        cv2.imwrite(f"train_{i}.jpg", image * 255.0)
        if i == 8:
            break

    for x_y in dataset.val_reader():
        image, label = x_y
        print("val image {}".format(image.shape))
        image = np.array(image).transpose((1, 2, 0))
        show_image(image, label)
        cv2.imwrite("val.jpg", image * 255.0)
        break

    for x_y in dataset.test_reader():
        image,path = x_y
        print("test image {}".format(image.shape))
        image = np.array(image).transpose((1, 2, 0))
        show_image(image, " ")
        break
