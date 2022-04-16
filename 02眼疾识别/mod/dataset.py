# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-16 11:37
文档说明: ImageClass 图像分类数据集解析
"""


from typing import Any
import paddle
import os
import struct
import numpy as np


class ImageClass(paddle.io.Dataset):
    """
    ImageClass 图像分类数据集解析, 继承 paddle.io.Dataset 类
    """

    def __init__(self,
                 dataset_path: str,
                 images_labels_txt_path: str,
                 transform=None,
                 ):
        """
        构造函数，定义数据集大小

        Args:
            dataset_path (str): 数据集路径
            images_labels_txt_path (str): 图像和标签的文本路径
            transform (Compose, optional): 转换数据的操作组合, 默认 None
        """
        super(ImageClass, self).__init__()
        self.dataset_path = dataset_path
        self.images_labels_txt_path = images_labels_txt_path
        self._check_path(dataset_path, "数据集路径错误")
        self._check_path(images_labels_txt_path, "图像和标签的文本路径错误")
        self.transform = transform
        self.images, self.labels = self.parse_dataset(dataset_path, images_labels_txt_path)

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
    def parse_dataset(dataset_path: str, images_labels_txt_path: str):
        """
        数据集解析

        Args:
            dataset_path (str): 数据集路径
            images_labels_txt_path (str): 图像和标签的文本路径

        Returns:
            images: 图像集
            labels: 标签集
        """
        lines = []
        images = []
        labels = []
        with open(images_labels_txt_path, "r") as f:
            lines = f.readlines()
        for i in lines:
            data = i.split(" ")
            images.append(os.path.join(dataset_path, data[0]))
            labels.append(int(data[1]))
        return images, labels
