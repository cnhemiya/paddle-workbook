# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-01 17:44
文档说明: GoogLeNet 网络模型
"""


import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Inception(nn.Layer):
    """
    Inception模块
    """

    def __init__(self, c0, c1, c2, c3, c4):
        """
        Inception模块

        Args:
            c0 (int): 模块输入通道数
            c1 (int): 第一支路 1x1 卷积输出通道数
            c2 (list | tuple): 第二支路 1x1 卷积输入通道数 , 3x3 卷积输出通道数
            c3 (list | tuple): 第三支路 1x1 卷积输入通道数 , 5x5 卷积输出通道数
            c4 (int): 第四支路 3x3 池化 , 1x1 卷积输出通道数
        """
        super(Inception, self).__init__()

        # 第一支路 1x1 卷积输出通道数
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=c0, out_channels=c1,
                      kernel_size=1, stride=1),
            nn.ReLU())

        # 第二支路 1x1 卷积输入通道数 , 3x3 卷积输出通道数
        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=c0,
                      out_channels=c2[0], kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=c2[0], out_channels=c2[1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        # 第三支路 1x1 卷积输入通道数 , 5x5 卷积输出通道数
        self.block3 = nn.Sequential(
            nn.Conv2D(in_channels=c0,
                      out_channels=c3[0], kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=c3[0], out_channels=c3[1],
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU())

        # 第四支路 3x3 池化 , 1x1 卷积输出通道数
        self.block4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=c0, out_channels=c4,
                      kernel_size=1, stride=1),
            nn.ReLU())

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(x)
        b3 = self.block3(x)
        b4 = self.block4(x)
        out = [b1, b2, b3, b4]
        return paddle.concat(out, axis=1)


# GoogLeNet 网络模型
class GoogLeNet(nn.Layer):
    """
    GoogLeNet 网络模型

    输入图像大小为 224 x 224
    """

    def __init__(self, num_classes=10):
        """
        GoogLeNet 网络模型

        Args:
            num_classes (int, optional): 分类数量, 默认 10

        Raises:
            Exception: 分类数量 num_classes 必须大于等于 2
        """
        super(GoogLeNet, self).__init__()
        if num_classes < 2:
            raise Exception(
                "分类数量 num_classes 必须大于等于 2: {}".format(num_classes))
        self.num_classes = num_classes

        # 数据输入处理块
        self.block0 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64,
                      kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=192,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1))

        # 第三个模块包含2个Inception块
        self.block3_1 = Inception(192, 64, [96, 128], [16, 32], 32)
        self.block3_2 = Inception(256, 128, [128, 192], [32, 96], 64)
        self.pool3 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第四个模块包含5个Inception块
        self.block4_1 = Inception(480, 192, [96, 208], [16, 48], 64)
        self.block4_2 = Inception(512, 160, [112, 224], [24, 64], 64)
        self.block4_3 = Inception(512, 128, [128, 256], [24, 64], 64)
        self.block4_4 = Inception(512, 112, [144, 288], [32, 64], 64)
        self.block4_5 = Inception(528, 256, [160, 320], [32, 128], 128)
        self.pool4 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第五个模块包含2个Inception块
        self.block5_1 = Inception(832, 256, [160, 320], [32, 128], 128)
        self.block5_2 = Inception(832, 384, [192, 384], [48, 128], 128)
        self.avg_pool5 = nn.AvgPool2D(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc5 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        # 数据输入处理块
        x = self.block0(x)

        # # 第三个模块包含2个Inception块
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.pool3(x)

        # # # 第四个模块包含5个Inception块
        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)
        x = self.block4_4(x)
        x = self.block4_5(x)
        x = self.pool4(x)

        # # 第五个模块包含2个Inception块
        x = self.block5_1(x)
        x = self.block5_2(x)
        x = self.avg_pool5(x)
        # flatten 根据给定的 start_axis 和 stop_axis 将连续的维度展平
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.dropout(x)
        x = self.fc5(x)
        x = F.softmax(x)

        return x
