# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-08 21:52
文档说明: 杂项
"""


import os
import time
import random
import paddle.nn.functional as F


def check_path(path: str, msg="路径错误"):
    """
    检查路径是否存在

    Args:
        path (str): 路径
        msg (str, optional): 异常消息, 默认 "路径错误"

    Raises:
        Exception: 路径错误, 异常
    """
    if not os.path.exists(path):
        raise Exception("{}: {}".format(msg, path))


def time_id(format="%Y-%m-%d_%H-%M-%S"):
    """
    根据时间生成的字符串 ID

    Args:
        format (str, optional): 格式化, 默认 "%Y-%m-%d_%H-%M-%S"

    Returns:
        str: 根据时间生成的字符串 ID
    """
    return time.strftime(format)


def predict_to_class(predict_result):
    """
    预测转分类标签

    Args:
        predict_result (tensor): tensor 数据

    Returns:
        int: 分类标签 id
    """
    result_list = F.softmax(predict_result[0]).tolist()
    result_idx = result_list.index(max(result_list))
    return result_idx


def parse_dataset(dataset_path: str, dataset_list_path: str, shuffle: bool):
    """
    数据集解析

    Args:
        dataset_path (str): 数据集目录路径
        dataset_list_path (str): 数据集列表文件路径

    Returns:
        image_paths: 图像路径集
        labels: 分类标签集
    """
    lines = []
    image_paths = []
    labels = []
    with open(dataset_list_path, "r") as f:
        lines = f.readlines()
    # 随机打乱数据
    if (shuffle):
        random.shuffle(lines)
    for i in lines:
        data = i.split(" ")
        if (len(data) < 2):
            raise Exception("数据集解析错误，数据少于 2")
        image_paths.append(os.path.join(dataset_path, data[0]))
        labels.append(int(data[1]))
    return image_paths, labels
