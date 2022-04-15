# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-08 21:52
文档说明: MNIST 手写数据集解析
"""


import paddle
import os
import struct
import numpy as np


class MNIST(paddle.io.Dataset):
    """
    MNIST 手写数据集解析, 继承 paddle.io.Dataset 类
    """

    def __init__(self,
                 images_path: str,
                 labels_path: str,
                 transform=None,
                 ):
        """
        构造函数，定义数据集大小

        Args:
            images_path (str): 图像集路径
            labels_path (str): 标签集路径
            transform (Compose, optional): 转换数据的操作组合, 默认 None
        """
        super(MNIST, self).__init__()
        self.images_path = images_path
        self.labels_path = labels_path
        self._check_path(images_path, "数据路径错误")
        self._check_path(labels_path, "标签路径错误")
        self.transform = transform
        self.images, self.labels = self.parse_dataset(images_path, labels_path)

    def __getitem__(self, idx):
        """
        获取单个数据和标签

        Args:
            idx (Any): 索引

        Returns:
            image (float32): 图像
            label (int64): 标签
        """
        image, label = self.images[idx], self.labels[idx]
        # 这里 reshape 是2维 [28 ,28]
        image = np.reshape(image, [28, 28])
        if self.transform is not None:
            image = self.transform(image)
        # label.astype 如果是整型，只能是 int64
        return image.astype('float32'), label.astype('int64')

    def __len__(self):
        """
        数据数量

        Returns:
            int: 数据数量
        """
        return len(self.labels)

    def _check_path(self, path: str, msg: str):
        """
        检查路径是否存在

        Args:
            path (str): 路径
            msg (str, optional): 异常消息

        Raises:
            Exception: 路径错误, 异常
        """
        if not os.path.exists(path):
            raise Exception("{}: {}".format(msg, path))

    @staticmethod
    def parse_dataset(images_path: str, labels_path: str):
        """
        数据集解析

        Args:
            images_path (str): 图像集路径
            labels_path (str): 标签集路径

        Returns:
            images: 图像集
            labels: 标签集
        """
        with open(images_path, 'rb') as imgpath:
            # 解析图像集
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            # 这里 reshape 是1维 [786]
            images = np.fromfile(
                imgpath, dtype=np.uint8).reshape(num, rows * cols)
        with open(labels_path, 'rb') as lbpath:
            # 解析标签集
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        return images, labels
