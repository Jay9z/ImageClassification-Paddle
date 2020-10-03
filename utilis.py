#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("training_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def show_image(image,title):
    image = np.array(image).transpose((1,2,0))
    plt.imshow(image)
    plt.title("cat {}".format(title))

# def print_model(x,m):
#     for item in m.sublayers():
#     # item是LeNet类中的一个子层
#     # 查看经过子层之后的输出数据形状
#         try:
#             x = item(x)
#         except:
#             x = fluid.layers.reshape(x, [x.shape[0], -1])
#             x = item(x)
#         if len(item.parameters())==2:
#             # 查看卷积和全连接层的数据和参数的形状，
#             # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
#             print(item.full_name(), x.shape, item.parameters()[0].shape, item.parameters()[1].shape)
#         else:
#             # 池化层没有参数
#             print(item.full_name(), x.shape)

def print_model(m):
    for item in m.sublayers():
        if len(item.parameters())==2:
            # 查看卷积和全连接层的数据和参数的形状，
            # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
            logger.info(item.full_name(),"weight shape: ", item.parameters()[0].shape,"bias shape: ", item.parameters()[1].shape)
            logger.info(item.full_name(),"weight: \r", item.parameters()[0].numpy())
            logger.info(item.full_name(),"bias: \r", item.parameters()[1].numpy())
        else:
            # 池化层没有参数
            print(item.full_name())