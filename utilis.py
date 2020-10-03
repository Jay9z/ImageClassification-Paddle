#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
if not os.path.exists("result"):
    os.makedirs("result")
handler = logging.FileHandler("result/training_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def show_image(image, title):
    plt.imshow(image)
    plt.title("cat {}".format(title))


def print_model(m):
    for item in m.sublayers():
        if len(item.parameters()) == 2:
            # 查看卷积和全连接层的数据和参数的形状，
            # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
            logger.info(
                item.full_name(),
                "weight shape: ",
                item.parameters()[0].shape,
                "bias shape: ",
                item.parameters()[1].shape,
            )
            logger.info(item.full_name(), "weight: \r", item.parameters()[0].numpy())
            logger.info(item.full_name(), "bias: \r", item.parameters()[1].numpy())
        else:
            # 池化层没有参数
            print(item.full_name())
