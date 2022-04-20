#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-09 22:57
文档说明: 查看报表
"""


import os
import sys
import shutil
import mod.report
import mod.config as config


def get_print_str(report: mod.report.Report):
    """
    获取打印 Report 的字符串
    """
    return "id: {},  loss: {:<.19f},  acc: {:<.4f},  EP: {},  BS: {},  LR: {}".format(
        report.id, report.loss, report.acc, report.epochs, report.batch_size, report.learning_rate)


def get_first_str():
    return "EP = epochs,  BS = batch_size,  LR = learning_rate\n"


def save_best(report: mod.report.Report):
    """
    保存最佳参数结果到 best 目录
    """
    if not os.path.exists(config.SAVE_BEST_PATH):
        os.mkdir(config.SAVE_BEST_PATH)
    old_dir = os.path.join(config.SAVE_DIR, report.id)
    files = os.listdir(old_dir)
    for file in files:
        if not os.path.isfile(os.path.join(old_dir, file)):
            continue
        src_file = os.path.join(old_dir, file)
        dst_file = os.path.join(config.SAVE_BEST_PATH, file)
        shutil.copyfile(src_file, dst_file)


def print_report():
    """
    打印 Report 列表, 返回最佳结果，没有最佳返回 None
    """
    dirs = os.listdir(config.SAVE_DIR)
    report_list = []
    for dir in dirs:
        sub_dir = os.path.join(config.SAVE_DIR, dir)
        if (not os.path.isdir(sub_dir)) or (dir == config.SAVE_BAST_DIR):
            continue
        report_file = os.path.join(sub_dir, config.REPORT_FILE)
        if not os.path.exists(report_file):
            continue
        report = mod.report.Report()
        report.load(report_file)
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
    if (best != None) and (len(sys.argv) > 1) and (sys.argv[1] == "--best"):
        print("保存模型参数。。。")
        save_best(best)
        print("模型参数保存完毕！")
    else:
        print("加参数 --best 保存最佳模型参数到 best 目录: report.py --best")


if __name__ == "__main__":
    main()
