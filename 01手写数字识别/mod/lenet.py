# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-08 21:52
文档说明: LeNet 网络模型
"""


import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# LeNet 网络模型
class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        if num_classes < 1:
            raise Exception("分类数量 num_classes 必须大于 0: {}".format(num_classes))
        self.num_classes = num_classes
        self.conv1 = nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.avg_pool1 = nn.AvgPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(
            in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2D(
            in_channels=16, out_channels=120, kernel_size=4, stride=1)
        self.fc1 = nn.Linear(in_features=120, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avg_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avg_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# 使用 Sequential 顺序组网
class LeNet_SEQ(nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        if num_classes < 1:
            raise Exception("分类数量 num_classes 必须大于 0: {}".format(num_classes))
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1D(kernel_size=2,  stride=2),
            nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1D(kernel_size=2, stride=2),
            nn.Conv2D(in_channels=16, out_channels=120, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return x
