# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-06 22:41
文档说明: ResNet 网络模型
"""


import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ResNetSimpleBlock(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, stride: 2, sample_stride: 2, is_sample: False):
        """_summary_

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            stride (2): _description_
            sample_stride (2): _description_
            is_sample (False): _description_
        """
        super(ResNetSimpleBlock, self).__init__()

        # 残差块
        self.base_block = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm(),
            nn.ReLU(),
            nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm()
        )

        self.is_sample = is_sample
        if (is_sample):
            self.sample_block = nn.Sequential(
                nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=sample_stride, padding=0),
                nn.BatchNorm()
            )

    def forward(self, x):
        residual = x
        y = self.base_block(x)
        if (self.is_sample):
            residual = self.sample_block(x)
        x = residual + y
        x = F.relu(x)
        return x


class ResNetBaseBlock(nn.Layer):
    """
    ResNetBlock 模块
    """

    def __init__(self, in_channels: int, out_channels: int, stride: 2, sample_stride: 2, is_sample: False):
        """_summary_

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            stride (2): _description_
            sample_stride (2): _description_
            is_sample (False): _description_
        """
        super(ResNetBaseBlock, self).__init__()

        # 残差块
        self.base_block = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(),
            nn.ReLU(),
            nn.Conv2D(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm(),
            nn.ReLU(),
            nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm()
        )

        self.is_sample = is_sample
        if (is_sample):
            self.sample_block = nn.Sequential(
                nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=sample_stride, padding=0),
                nn.BatchNorm()
            )

    def forward(self, x):
        residual = x
        y = self.base_block(x)
        if (self.is_sample):
            residual = self.sample_block(x)
        x = residual + y
        x = F.relu(x)
        return x


class ResNet(nn.Layer):
    def __init__(self, net_block, blocks, num_classes=10):
        super(ResNet, self).__init__()
        if num_classes < 2:
            raise Exception(
                "分类数量 num_classes 必须大于等于 2: {}".format(num_classes))
        self.num_classes = num_classes

        self.in_block = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm(),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        )

        self.avg_pool = nn.AvgPool2D(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.fc(x)
        x = F.softmax(x)
        return x


def resnet18(num_classes: int):
    return ResNet(ResNetSimpleBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes: int):
    return ResNet(ResNetSimpleBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes: int):
    return ResNet(ResNetBaseBlock, [3, 4, 6, 3], num_classes)


def resnet101(num_classes: int):
    return ResNet(ResNetBaseBlock, [3, 4, 23, 3], num_classes)


def resnet152(num_classes: int):
    return ResNet(ResNetBaseBlock, [3, 8, 36, 3], num_classes)
