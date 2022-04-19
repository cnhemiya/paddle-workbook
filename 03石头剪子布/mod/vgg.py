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
                根据 max_pool3 输出结果, 计算得出 512*7*7 = 25088

        Raises:
            Exception: 分类数量 num_classes 必须大于等于 2
        """
        super(VGG, self).__init__()
        if num_classes < 2:
            raise Exception(
                "分类数量 num_classes 必须大于等于 2: {}".format(num_classes))
        self.num_classes = num_classes
        self.fc1_in_features = fc1_in_features

    def forward(self, x):
        x = 0
        return x
