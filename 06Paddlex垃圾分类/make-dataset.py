#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-01 18:38
"""


import os
import sys


TRAIN_LIST_FILE = "train_list.txt"
TEST_LIST_FILE = "test_list.txt"


MAKE_ALL = "all"
MAKE_TRAIN = "train"
MAKE_TEST = "test"


TRAIN_SIZE_RATIO = 0.8


HELP_DOC = """
使用示例：

make-dataset.py all ./dataset img_dir1 0 img_dir2 1 img_dir3 2

all: 生成图像标签文本, all 生成 train 和 test, train 生成 train, test 生成 test
./dataset: 数据集文件夹
img_dir: dataset 目录下的文件夹
0 1 2: 是图像文件夹对应的分类标签
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


def make_dataset(make: str, dataset: str, dataset_list, train_file: str, test_file: str):
    if make not in ["all", "train","test"]:
        raise Exception(
                "生成图像标签文本参数错误，只能是 all, train, test, 接收参数为: {}".format(make))
    train_data = []
    test_data = []
    for i in dataset_list:
        data = dataset_list_data(
            dataset_dir=dataset, image_dir=i[0], label=int(i[1]))
        if make == MAKE_ALL:
            train_size = int(len(data) * TRAIN_SIZE_RATIO)
            train_data.extend(data[:train_size])
            test_data.extend(data[train_size:])
        elif make == MAKE_TRAIN:
            train_data.extend(data)
        elif make == MAKE_TEST:
            test_data.extend(data)
    if make == MAKE_ALL or make == MAKE_TRAIN:
        write_file(os.path.join(dataset, train_file), train_data)
    if make == MAKE_ALL or make == MAKE_TEST:
        write_file(os.path.join(dataset, test_file), test_data)


def main():
    if (len(sys.argv) < 4):
        print(HELP_DOC)
    else:
        make, dataset, dataset_list = arg_parse()
        make_dataset(make=make, dataset=dataset, dataset_list=dataset_list,
                           train_file=TRAIN_LIST_FILE, test_file=TEST_LIST_FILE)
        if make == MAKE_ALL or make == MAKE_TRAIN:                   
            print("生成训练集: {}".format(os.path.join(dataset, TRAIN_LIST_FILE)))
        if make == MAKE_ALL or make == MAKE_TEST:
            print("生成测试集: {}".format(os.path.join(dataset, TEST_LIST_FILE)))


if __name__ == "__main__":
    main()
