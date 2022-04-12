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
    步骤一: 继承paddle.io.Dataset类
    """

    def __init__(self,
                 image_path=None,
                 label_path=None,
                 transform=None,
                 ):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MNIST, self).__init__()
        self.image_path = image_path
        self.label_path = label_path
        self._check_path(image_path, "数据路径错误")
        self._check_path(label_path, "标签路径错误")
        self.transform = transform
        self.images, self.labels = self.parse_dataset(image_path, label_path)

    def __getitem__(self, idx):
        """
        获取单个数据和标签
        """
        image, label = self.images[idx], self.labels[idx]
        # 这里 reshape 是2维 [28 ,28]
        image = np.reshape(image, [28, 28])
        if self.transform is not None:
            image = self.transform(image)
        # label.astype 如果是整型，只能是 int64
        return image.astype('float32'), label.astype('int64')

    def __len__(self):
        return len(self.labels)

    def _check_path(self, path, msg):
        if not os.path.exists(path):
            raise Exception("{}: {}".format(msg, path))

    @staticmethod
    def parse_dataset(image_path, label_path):
        """
        数据集解析
        """
        with open(image_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            # 这里 reshape 是1维 [786]
            images = np.fromfile(
                imgpath, dtype=np.uint8).reshape(num, rows * cols)
        with open(label_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        return images, labels
