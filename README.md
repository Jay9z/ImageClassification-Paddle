Cats 

## Introduction
Demonstrate the workflow of do image classification task with paddle

## Pre-requisites
1. paddle 1.8
2. numpy
3. matplotlib
4. pandas
5. cats dataset
6. paddlex

## Installation
1. install packages
   ```
   pip install -r requirements
   pip install albumentations -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. download cats dataset
   ```
   链接：https://pan.baidu.com/s/1cE3AbX1UzDsbeD2pTbFSPw 
   提取码：i29d 
   ```

## How to use it

```
python train_predict.py
```

## Lesson learned

1. Comparing with tensorflow, it is quite convinent to integrate three party python packages with paddle framework.

2. sometimes, cv2.imread() may not work, but PIL.Image can find the file and open it.

3. paddlex.cls.transforms can be used for image augmentation.

4. pandas do a better job than numpy when saving data to file.

5. Model, Optimizer, and batch_size should be tweaked when loss can be optimized from the begining.
6. Excessive augmentation could lead loss exposure, and bigger batch_size could help with this problem.