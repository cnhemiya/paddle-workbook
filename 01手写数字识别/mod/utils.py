# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-08 21:52
文档说明: 杂项
"""


import os
import time


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


def time_str(format="%Y-%m-%d_%H-%M-%S"):
    """
    时间格式化字符串

    Args:
        format (str, optional): 格式化, 默认 "%Y-%m-%d_%H-%M-%S"

    Returns:
        str: 时间格式化字符串
    """    
    return time.strftime(format)
