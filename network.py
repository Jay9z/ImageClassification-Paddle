#!/usr/bin/env python
# coding: utf-8
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, BatchNorm


# 定义 LeNet 网络结构
class LeNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=12):
        super(LeNet, self).__init__()

        # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        self.conv1 = Conv2D(
            num_channels=3, num_filters=6, filter_size=5, act="sigmoid"
        )  # 1-->3 for color image
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type="max")
        self.conv2 = Conv2D(
            num_channels=6, num_filters=16, filter_size=5, act="sigmoid"
        )
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type="max")
        # 创建第3个卷积层
        self.conv3 = Conv2D(
            num_channels=16, num_filters=120, filter_size=4, act="sigmoid"
        )
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc1 = Linear(
            input_dim=300000, output_dim=64, act="sigmoid"
        )  # 120-->300000 ## bigger image 28*28-->224*224
        self.fc2 = Linear(input_dim=64, output_dim=num_classes)

    # 网络的前向计算过程
    def forward(self, input, label=None):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class VGG16(fluid.dygraph.Layer):
    def __init__(self, num_classes=12):
        super(VGG16, self).__init__()
        self.block1_conv1_3_64 = ConvBNLayer(
            num_channels=3, num_filters=64, filter_size=3, act="relu"
        )
        self.block1_conv2_3_64 = ConvBNLayer(
            num_channels=64, num_filters=64, filter_size=3, act="relu"
        )
        self.block1_maxpool1 = fluid.dygraph.Pool2D(
            pool_size=(2, 2), pool_stride=2, pool_padding=0, pool_type="max"
        )
        self.block2_conv1_3_128 = ConvBNLayer(
            num_channels=64, num_filters=128, filter_size=3, act="relu"
        )
        self.block2_conv2_3_128 = ConvBNLayer(
            num_channels=128, num_filters=128, filter_size=3, act="relu"
        )
        self.block2_maxpool1 = fluid.dygraph.Pool2D(
            pool_size=(2, 2), pool_stride=2, pool_padding=0, pool_type="max"
        )
        self.block3_conv1_3_256 = ConvBNLayer(
            num_channels=128, num_filters=256, filter_size=3, act="relu"
        )
        self.block3_conv2_3_256 = ConvBNLayer(
            num_channels=256, num_filters=256, filter_size=3, act="relu"
        )
        self.block3_conv3_3_256 = ConvBNLayer(
            num_channels=256, num_filters=256, filter_size=3, act="relu"
        )
        self.block3_maxpool1 = fluid.dygraph.Pool2D(
            pool_size=(2, 2), pool_stride=2, pool_padding=0, pool_type="max"
        )
        self.block4_conv1_3_512 = ConvBNLayer(
            num_channels=256, num_filters=512, filter_size=3, act="relu"
        )
        self.block4_conv2_3_512 = ConvBNLayer(
            num_channels=512, num_filters=512, filter_size=3, act="relu"
        )
        self.block4_conv3_3_512 = ConvBNLayer(
            num_channels=512, num_filters=512, filter_size=3, act="relu"
        )
        self.block4_maxpool1 = fluid.dygraph.Pool2D(
            pool_size=(2, 2), pool_stride=2, pool_padding=0, pool_type="max"
        )
        self.block5_conv1_3_512 = ConvBNLayer(
            num_channels=512, num_filters=512, filter_size=3, act="relu"
        )
        self.block5_conv2_3_512 = ConvBNLayer(
            num_channels=512, num_filters=512, filter_size=3, act="relu"
        )
        self.block5_conv3_3_512 = ConvBNLayer(
            num_channels=512, num_filters=512, filter_size=3, act="relu"
        )
        self.block5_maxpool1 = fluid.dygraph.Pool2D(
            global_pooling=True, pool_type="max"
        )  # 全局池化层
        self.fc1 = fluid.dygraph.Linear(input_dim=512, output_dim=num_classes)

    def forward(self, input, label=None):
        x = self.block1_conv1_3_64(input)
        x = self.block1_conv2_3_64(x)
        x = self.block1_maxpool1(x)
        x = self.block2_conv1_3_128(x)
        x = self.block2_conv2_3_128(x)
        x = self.block2_maxpool1(x)
        x = self.block3_conv1_3_256(x)
        x = self.block3_conv2_3_256(x)
        x = self.block3_conv3_3_256(x)
        x = self.block3_maxpool1(x)
        x = self.block4_conv1_3_512(x)
        x = self.block4_conv2_3_512(x)
        x = self.block4_conv3_3_512(x)
        x = self.block4_maxpool1(x)
        x = self.block5_conv1_3_512(x)
        x = self.block5_conv2_3_512(x)
        x = self.block5_conv3_3_512(x)
        x = self.block5_maxpool1(x)
        x = fluid.layers.squeeze(x, axes=[])  # 多余的维度去除(32,1,1,512)-->(32,512)
        x = self.fc1(x)
        return x


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(
        self, num_channels, num_filters, filter_size, stride=1, groups=1, act=None
    ):
        """

        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        act, 激活函数类型，默认act=None不使用激活函数
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
        )

        # 创建BatchNorm层
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu",
        )
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu",
        )
        # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
        )

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride,
            )

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2, act="relu")
        return y


# 定义ResNet模型
class ResNet(fluid.dygraph.Layer):
    def __init__(self, layers=50, num_classes=12):
        """

        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert (
            layers in supported_layers
        ), "supported layers are {} but input layer is {}".format(
            supported_layers, layers
        )

        if layers == 50:
            # ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            # ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            # ResNet152包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]

        # 残差块中使用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3, num_filters=64, filter_size=7, stride=2, act="relu"
        )
        self.pool2d_max = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type="max"
        )

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    "bb_%d_%d" % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2
                        if i == 0 and block != 0
                        else 1,  # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                        shortcut=shortcut,
                    ),
                )
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = Pool2D(pool_size=7, pool_type="avg", global_pooling=True)

        # stdv用来作为全连接层随机初始化参数的方差
        import math

        stdv = 1.0 / math.sqrt(2048 * 1.0)

        # 创建全连接层，输出大小为类别数目
        self.out = Linear(
            input_dim=2048,
            output_dim=num_classes,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)
            ),
        )

    def forward(self, inputs, label=None):
        x = self.conv(inputs)
        x = self.pool2d_max(x)
        for bottleneck_block in self.bottleneck_block_list:
            x = bottleneck_block(x)
        x = self.pool2d_avg(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.out(x)
        return x
