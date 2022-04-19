# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-19 11:33
文档说明: VGG 网络模型
"""


import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# VGG 网络模型
class VGG(nn.Layer):
    """
    VGG 网络模型

    输入图像大小为 224 x 224
    """

    def __init__(self, num_classes=10, fc1_in_features=25088):
        """
        VGG 网络模型

        Args:
            num_classes (int, optional): 分类数量, 默认 10
            fc1_in_features (int, optional): 第一层全连接层输入特征数量, 默认 25088, 
                根据 max_pool5 输出结果, 计算得出 512*7*7 = 25088

        Raises:
            Exception: 分类数量 num_classes 必须大于等于 2
        """
        super(VGG, self).__init__()
        if num_classes < 2:
            raise Exception(
                "分类数量 num_classes 必须大于等于 2: {}".format(num_classes))
        self.num_classes = num_classes
        self.fc1_in_features = fc1_in_features

        # 处理块 1
        self.conv1_1 = nn.Conv2D(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2D(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 处理块 2
        self.conv2_1 = nn.Conv2D(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2D(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 处理块 3
        self.conv3_1 = nn.Conv2D(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2D(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2D(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 处理块 4
        self.conv4_1 = nn.Conv2D(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.max_pool4 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 处理块 5
        self.conv5_1 = nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.max_pool5 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 全连接层 in_features 25088 = max_pool5 输出 512*7*7
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=4096)
        self.drop_ratio1 = 0.5
        self.drop1 = nn.Dropout(self.drop_ratio1)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.drop_ratio2 = 0.5
        self.drop2 = nn.Dropout(self.drop_ratio2)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        # 处理块 1
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.max_pool1(x)

        # 处理块 2
        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.max_pool2(x)

        # 处理块 3
        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = F.relu(x)
        x = self.max_pool3(x)

        # 处理块 4
        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = F.relu(x)
        x = self.max_pool4(x)

        # 处理块 5
        x = self.conv5_1(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = F.relu(x)
        x = self.max_pool5(x)

        # 全连接层
        # flatten 根据给定的 start_axis 和 stop_axis 将连续的维度展平
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc1(x)
        x = F.relu(x)
        # 在全连接之后使用 dropout 抑制过拟合
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x)
        # 在全连接之后使用 dropout 抑制过拟合
        x = self.drop2(x)
        x = self.fc3(x)

        return x
