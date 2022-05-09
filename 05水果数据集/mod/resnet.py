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

        self.is_sample = is_sample  # 是否采样模块
        self.is_simple = is_simple  # 是否简易模块

        in_channels = channels[0]   # 输入通道
        mid_channels = channels[1]  # 中间通道
        out_channels = channels[2]  # 输出通道

        # 残差模块
        self.block = nn.Sequential()
        if (is_simple):
            # 简易模块
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
            # 正常模块
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
            # 采样模块
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
    """
    ResNet 网络模型

    输入图像大小为 224 x 224
    """

    def __init__(self, blocks, num_classes=10, is_simple=False):
        """
        ResNet 网络模型

        Args:
            blocks (list|tuple): 每模块数量
            num_classes (int, optional): 分类数量, 默认 10
            is_simple (bool, optional): 是否简易模块，默认 False, 默认 不是简易模块

        Raises:
            Exception: 分类数量 num_classes < 2
        """
        super(ResNet, self).__init__()
        if num_classes < 2:
            raise Exception(
                "分类数量 num_classes 必须大于等于 2: {}".format(num_classes))

        self.num_classes = num_classes  # 分类数量
        self.is_simple = is_simple  # 是否简易模块

        # 简易模块通道, [0输入通道, 1中间通道, 2输出通道]
        self.simple_channels = [[64, 64, 128],
                                [128, 128, 256],
                                [256, 256, 512],
                                [512, 512, 512]]

        # 正常模块通道, [0输入通道, 1中间通道, 2输出通道]
        self.base_channels = [[64, 64, 256],
                              [256, 128, 512],
                              [512, 256, 1024],
                              [1024, 512, 2048]]

        # 输入模块
        self.in_block = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm(num_channels=64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        )

        # 处理模块
        self.block = self.make_blocks(blocks)

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

    def make_blocks(self, blocks):
        """
        生成所有模块

        Args:
            blocks (list|tuple): 每模块数量

        Returns:
            paddle.nn.Sequential: 所有模块顺序连接
        """
        seq = []
        is_in_block = True
        for block_index in range(len(blocks)):
            is_first_block = True
            for i in range(blocks[block_index]):
                seq.append(self.make_one_block(block_index=block_index,
                                               is_in_block=is_in_block, is_first_block=is_first_block))
                is_first_block = False
            is_in_block = False
        return nn.Sequential(*seq)

    def make_one_block(self, block_index: int, is_in_block: bool, is_first_block: bool):
        """
        生成一个模块

        Args:
            block_index (int): 模块索引
            is_in_block (bool): 是否残差输入模块
            is_first_block (bool): 是否第一模块

        Returns:
            ResNetBlock: 残差模块
        """
        net = None
        stride = 1
        sample_stride = 2
        if is_in_block:
            stride = 1 if is_first_block else 1
            sample_stride = 1 if is_first_block else 2
        else:
            stride = 2 if is_first_block else 1
            sample_stride = 2
        channels1 = self.simple_channels[block_index] if self.is_simple else self.base_channels[block_index]
        if is_first_block:
            net = ResNetBlock(channels=channels1, stride=stride, sample_stride=sample_stride,
                              is_sample=is_first_block, is_simple=self.is_simple)
        else:
            channels2 = [channels1[2], channels1[1], channels1[2]]
            net = ResNetBlock(channels=channels2, stride=stride, sample_stride=sample_stride,
                              is_sample=is_first_block, is_simple=self.is_simple)
        return net


def get_resnet(num_classes: int, resnet=50):
    """
    获取 ResNet 网络模型

    Args:
        num_classes (int, optional): 分类数量
        resnet (int, optional): ResNet模型选项, 默认 50, 可选 18, 34, 50, 101, 152

    Returns:
        ResNet: ResNet 网络模型
    """
    if resnet not in [18, 34, 50, 101, 152]:
        raise Exception(
            "resnet 可选 18, 34, 50, 101, 152, 实际: {}".format(resnet))

    net = None
    if resnet == 18:
        net = ResNet([2, 2, 2, 2], num_classes, is_simple=True)
    elif resnet == 34:
        net = ResNet([3, 4, 6, 3], num_classes, is_simple=True)
    elif resnet == 50:
        net = ResNet([3, 4, 6, 3], num_classes, is_simple=False)
    elif resnet == 101:
        net = ResNet([3, 4, 23, 3], num_classes, is_simple=False)
    elif resnet == 152:
        net = ResNet([3, 8, 36, 3], num_classes, is_simple=False)

    return net
