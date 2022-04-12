# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-08 21:52
文档说明: 杂项
"""


import os
import time


def check_path(path, msg="路径错误"):
    """
    检查路径是否存在
    """
    if not os.path.exists(path):
        raise Exception("{}: {}".format(msg, path))


def time_str(format="%Y-%m-%d_%H-%M-%S"):
    """
    时间格式化字符串
    """
    return time.strftime(format)
