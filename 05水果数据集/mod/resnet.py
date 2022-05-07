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


class ResNetBlock(nn.Layer):
    """
    ResNetBlock 模块
    """

    def __init__(self, channels, stride=1, sample_stride=2, is_sample=False, is_simple=False):
        """
        ResNetBlock 模块

        Args:
            channels (list|tuple): 3个, 0输入通道, 1中间通道, 2输出通道
            stride (int, optional): 模块步幅，默认 1.
            sample_stride (int, optional): 采样模块步幅，默认 2
            is_sample (bool, optional): 是否采样模块，默认 False, 默认 不是采样模块
            is_simple (bool, optional): 是否简易模块，默认 False, 默认 不是简易模块
        """
        super(ResNetBlock, self).__init__()

        self.is_sample = is_sample
        self.is_simple = is_simple

        in_channels = channels[0]
        mid_channels = channels[1]
        out_channels = channels[2]

        # 残差块
        self.block = nn.Sequential()
        if (is_simple):
            self.block = nn.Sequential(
                nn.Conv2D(in_channels=in_channels, out_channels=mid_channels,
                          kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm(num_channels=mid_channels),
                nn.ReLU(),
                nn.Conv2D(in_channels=mid_channels, out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm(num_channels=out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2D(in_channels=in_channels, out_channels=mid_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm(num_channels=mid_channels),
                nn.ReLU(),
                nn.Conv2D(in_channels=mid_channels, out_channels=mid_channels,
                          kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm(num_channels=mid_channels),
                nn.ReLU(),
                nn.Conv2D(in_channels=mid_channels, out_channels=out_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm(num_channels=out_channels)
            )

        if (is_sample):
            self.sample_block = nn.Sequential(
                nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=sample_stride, padding=0),
                nn.BatchNorm(num_channels=out_channels)
            )

    def forward(self, x):
        residual = x
        y = self.block(x)
        if (self.is_sample):
            residual = self.sample_block(x)
        x = paddle.add(x=residual, y=y)
        x = F.relu(x)
        return x


class ResNet(nn.Layer):
    def __init__(self, blocks, num_classes=10, is_simple=False):
        super(ResNet, self).__init__()
        if num_classes < 2:
            raise Exception(
                "分类数量 num_classes 必须大于等于 2: {}".format(num_classes))
        self.num_classes = num_classes
        self.is_simple = is_simple

        self.simple_channels = [[64, 64, 128], [
            128, 128, 256], [256, 256, 512], [512, 512, 512]]
        self.base_channels = [[64, 64, 256], [256, 128, 512], [
            512, 256, 1024], [1024, 512, 2048]]

        # 输入模块
        self.in_block = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm(num_channels=64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        )

        # 处理模块
        self.block = self.make_block(blocks)

        # 输出模块
        self.avg_pool = nn.AvgPool2D(kernel_size=7, stride=1)
        self.features = 512 if is_simple else 2048
        self.fc = nn.Linear(self.features, num_classes)

    def forward(self, x):
        x = self.in_block(x)
        x = self.block(x)
        x = self.avg_pool(x)
        # flatten 根据给定的 start_axis 和 stop_axis 将连续的维度展平
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc(x)
        return x

    def make_block(self, blocks):
        seq = []
        is_in_block = True
        for blk_i in range(len(blocks)):
            is_first_block = True
            for i in range(blocks[blk_i]):
                stride = 1
                sample_stride = 2
                if is_in_block:
                    stride = 1 if is_first_block else 1
                    sample_stride = 1 if is_first_block else 2
                else:
                    stride = 2 if is_first_block else 1
                    sample_stride = 2
                channels1 = self.base_channels[blk_i]
                if is_first_block:
                    seq.append(ResNetBlock(channels=channels1, stride=stride, sample_stride=sample_stride,
                                           is_sample=is_first_block, is_simple=self.is_simple))
                else:
                    channels2 = [channels1[2], channels1[1], channels1[2]]
                    seq.append(ResNetBlock(channels=channels2, stride=stride, sample_stride=sample_stride,
                                           is_sample=is_first_block, is_simple=self.is_simple))
                is_first_block = False
            is_in_block = False
        return nn.Sequential(*seq)


# def resnet18(num_classes: int):
#     return ResNet([2, 2, 2, 2], num_classes, is_simple=True)


# def resnet34(num_classes: int):
#     return ResNet([3, 4, 6, 3], num_classes, is_simple=True)


def resnet50(num_classes: int):
    return ResNet([3, 4, 6, 3], num_classes, is_simple=False)


def resnet101(num_classes: int):
    return ResNet([3, 4, 23, 3], num_classes, is_simple=False)


def resnet152(num_classes: int):
    return ResNet([3, 8, 36, 3], num_classes, is_simple=False)
