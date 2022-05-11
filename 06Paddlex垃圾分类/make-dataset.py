#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-01 18:38
文档:     生成数据集列表
"""


import os
import sys


TRAIN_LIST_FILE = "train_list.txt"
EVAL_LIST_FILE = "val_list.txt"
TEST_LIST_FILE = "test_list.txt"


MAKE_ALL = "all"
MAKE_TRAIN = "train"
MAKE_EVAL = "eval"
MAKE_TEST = "test"


TRAIN_SIZE_RATE = 0.7
EVAL_SIZE_RATE = 0.2
TEST_SIZE_RATE = 0.1


HELP_DOC = """
使用示例：

make-dataset.py all ./dataset img_dir1 0 img_dir2 1 img_dir3 2

all: 生成数据集列表, all 生成 train, eval, test.
train 生成 train, eval 生成 eval, test 生成 test

./dataset: 数据集目录
img_dir: dataset 目录下的文件夹
0 1 2: 是图像目录对应的分类标签
"""


def dataset_list_data(dataset_dir: str, image_dir: str, label: int):
    image_dir_path = os.path.join(dataset_dir, image_dir)
    files = os.listdir(image_dir_path)
    lines = []
    for i in files:
        file_path = os.path.join(image_dir_path, i)
        if os.path.isdir(file_path):
            continue
        lines.append("{}/{} {}\n".format(image_dir, i, label))
    return lines


def write_file(file_name: str, lines):
    with open(file_name, "w") as f:
        f.writelines(lines)


def arg_parse():
    args = sys.argv
    make = args[1]
    dataset = args[2]
    args = args[3:]
    dataset_list = []
    for i in range(0, len(args), 2):
        idl = []
        idl.append(args[i])
        idl.append(args[i+1])
        dataset_list.append(idl)
    return make, dataset, dataset_list


def make_dataset(make: str, dataset: str, dataset_list, train_file: str, eval_file: str, test_file: str):
    if make not in ["all", "train", "eval", "test"]:
        raise Exception(
            "生成数据集列表参数错误，只能是 all, train, eval, test, 接收参数为: {}".format(make))
    train_data = []
    eval_data = []
    test_data = []
    for i in dataset_list:
        data = dataset_list_data(
            dataset_dir=dataset, image_dir=i[0], label=int(i[1]))
        if make == MAKE_ALL:
            train_size = int(len(data) * TRAIN_SIZE_RATE)
            eval_size = int(len(data) * EVAL_SIZE_RATE)
            train_data.extend(data[:train_size])
            eval_data.extend(data[train_size:train_size + eval_size])
            test_data.extend(data[train_size + eval_size:])
        elif make == MAKE_TRAIN:
            train_data.extend(data)
        elif make == MAKE_EVAL:
            eval_data.extend(data)
        elif make == MAKE_TEST:
            test_data.extend(data)
    if make == MAKE_ALL:
        write_file(os.path.join(dataset, train_file), train_data)
        write_file(os.path.join(dataset, eval_file), eval_data)
        write_file(os.path.join(dataset, test_file), test_data)
    elif make == MAKE_TRAIN:
        write_file(os.path.join(dataset, train_file), train_data)
    elif make == MAKE_EVAL:
        write_file(os.path.join(dataset, eval_file), eval_data)
    elif make == MAKE_TEST:
        write_file(os.path.join(dataset, test_file), test_data)


def main():
    if (len(sys.argv) < 4):
        print(HELP_DOC)
    else:
        make, dataset, dataset_list = arg_parse()
        make_dataset(make=make, dataset=dataset, dataset_list=dataset_list,
                     train_file=TRAIN_LIST_FILE, eval_file=EVAL_LIST_FILE, test_file=TEST_LIST_FILE)
        if make == MAKE_ALL or make == MAKE_TRAIN:
            print("生成训练集: {}".format(os.path.join(dataset, TRAIN_LIST_FILE)))
        if make == MAKE_ALL or make == MAKE_EVAL:
            print("生成评估集: {}".format(os.path.join(dataset, EVAL_LIST_FILE)))
        if make == MAKE_ALL or make == MAKE_TEST:
            print("生成测试集: {}".format(os.path.join(dataset, TEST_LIST_FILE)))


if __name__ == "__main__":
    main()
