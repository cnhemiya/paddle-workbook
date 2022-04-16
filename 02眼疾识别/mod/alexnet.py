# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-16 10:04
文档说明: AlexNet 网络模型
"""


import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# AlexNet 网络模型
class AlexNet(nn.Layer):
    """
    AlexNet 网络模型

    输入图像大小为 224 x 224
    池化层 kernel_size = 2, 第一层卷积层填充 paddling = 2
    """
    def __init__(self, num_classes=10, pool_kernel_size=2, conv1_paddling=2, fc1_in_features=9216):
        """
        AlexNet 网络模型

        Args:
            num_classes (int, optional): 分类数量, 默认 10
            pool_kernel_size (int, optional): 池化层核大小, 默认 2
            conv1_paddling (int, optional): 第一层卷积层填充, 默认 2,
                输入图像大小为 224 x 224 填充 2
            fc1_in_features (int, optional): 第一层全连接层输入特征数量, 默认 9216, 
                根据 max_pool3 输出结果, 计算得出 256*6*6 = 9216

        Raises:
            Exception: 分类数量 num_classes 必须大于等于 2
        """        
        super(AlexNet, self).__init__()
        if num_classes < 2:
            raise Exception("分类数量 num_classes 必须大于等于 2: {}".format(num_classes))
        self.num_classes = num_classes
        self.pool_kernel_size = pool_kernel_size
        self.fc1_in_features = fc1_in_features
        self.conv1 = nn.Conv2D(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=conv1_paddling)
        self.max_pool1 = nn.MaxPool2D(kernel_size=pool_kernel_size, stride=2)
        self.conv2 = nn.Conv2D(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.max_pool2 = nn.MaxPool2D(kernel_size=pool_kernel_size, stride=2)
        self.conv3 = nn.Conv2D(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2D(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2D(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2D(kernel_size=pool_kernel_size, stride=2)
        # in_features 9216 = max_pool3 输出 256*6*6
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=4096)
        self.drop_ratio1 = 0.5
        self.drop1 = nn.Dropout(self.drop_ratio1)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.drop_ratio2 = 0.5
        self.drop2 = nn.Dropout(self.drop_ratio2)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        # flatten 根据给定的start_axis 和 stop_axis 将连续的维度展平
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc1(x)
        x = F.relu(x)
        # 在全连接之后使用dropout抑制过拟合
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x)
        # 在全连接之后使用dropout抑制过拟合
        x = self.drop2(x)
        x = self.fc3(x)
        return x
