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


# GoogLeNet 网络模型
class GoogLeNet(nn.Layer):
    """
    GoogLeNet 网络模型

    输入图像大小为 224 x 224
    """

    def __init__(self, num_classes=10, fc1_in_features=25088):
        """
        GoogLeNet 网络模型

        Args:
            num_classes (int, optional): 分类数量, 默认 10
            fc1_in_features (int, optional): 第一层全连接层输入特征数量, 默认 25088, 
                根据 max_pool5 输出结果, 计算得出 512*7*7 = 25088

        Raises:
            Exception: 分类数量 num_classes 必须大于等于 2
        """
        super(GoogLeNet, self).__init__()
        if num_classes < 2:
            raise Exception(
                "分类数量 num_classes 必须大于等于 2: {}".format(num_classes))
        self.num_classes = num_classes
        self.fc1_in_features = fc1_in_features

    def forward(self, x):
        return x
