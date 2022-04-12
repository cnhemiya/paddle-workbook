#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-09 22:57
文档说明: 查看报表
"""


import os
import shutil
import mod.report


# 报表基本路径
BASE_PATH = "./params/"
# 最佳参数结果目录
BEST_DIR = "best"
# 最佳参数结果路径
BEST_PATH = BASE_PATH + BEST_DIR + "/"
# 报表文件名
REPORT_FILE = "report.json"


def get_print_str(report: mod.report.Report):
    """
    获取打印 Report 的字符串
    """
    return "id: {},  loss: {:<.19f},  acc: {},  EP: {},  BS: {},  LR: {}".format(
        report.id, report.loss, report.acc, report.epochs, report.batch_size, report.learning_rate)

def get_first_str():
    return "EP = epochs,  BS = batch_size,  LR = learning_rate\n"

def save_best(report: mod.report.Report):
    """
    保存最佳参数结果到 best 目录
    """
    if not os.path.exists(BEST_PATH):
        os.mkdir(BEST_PATH)
    old_dir = BASE_PATH + report.id
    files = os.listdir(old_dir)
    for file in files:
        if not os.path.isfile(os.path.join(old_dir, file)):
            continue
        src_file = os.path.join(old_dir, file)
        dst_file = os.path.join(BEST_PATH, file)
        shutil.copyfile(src_file, dst_file)


def print_report():
    """
    打印 Report 列表, 返回最佳结果，没有最佳返回 None
    """
    dirs = os.listdir(BASE_PATH)
    report_list = []
    for dir in dirs:
        if (not os.path.isdir(BASE_PATH + dir)) or (dir == BEST_DIR):
            continue
        report = mod.report.Report()
        report.load(os.path.join(BASE_PATH + dir, REPORT_FILE))
        report_list.append(report)
    best = None
    if (len(report_list) > 0):
        sort_list = sorted(report_list, key=lambda x: x.loss)
        print(get_first_str())
        for i in sort_list:
            print(get_print_str(i))
        best = sort_list[0]
        print("\nbest:  " + get_print_str(best))
    return best


def main():
    best = print_report()
    if (best != None):
        save_best(best)


if __name__ == "__main__":
    main()
