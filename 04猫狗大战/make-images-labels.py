#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-01 18:38
"""


import os
import sys


TRAIN_TXT_FILE = "train-images-labels.txt"
TEST_TXT_FILE = "test-images-labels.txt"


MAKE_ALL = "all"
MAKE_TRAIN = "train"
MAKE_TEST = "test"


TRAIN_SIZE_RATIO = 0.8


HELP_DOC = """
使用示例：

make-images-labels.py all ./dataset img_dir1 0 img_dir2 1 img_dir3 2

all: 生成图像标签文本, all 生成 train 和 test, train 生成 train, test 生成 test
./dataset: 数据集文件夹
img_dir: dataset 目录下的文件夹
0 1 2: 是图像文件夹对应的分类标签
"""


def images_lables_data(dataset_dir: str, image_dir: str, label: int):
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
    image_dir_label = []
    for i in range(0, len(args), 2):
        idl = []
        idl.append(args[i])
        idl.append(args[i+1])
        image_dir_label.append(idl)
    return make, dataset, image_dir_label


def make_images_labels(make: str, dataset: str, image_dir_label, train_file: str, test_file: str):
    if make not in ["all", "train","test"]:
        raise Exception(
                "生成图像标签文本参数错误，只能是 all, train, test, 接收参数为: {}".format(make))
    train_data = []
    test_data = []
    for i in image_dir_label:
        data = images_lables_data(
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
        make, dataset, image_dir_label = arg_parse()
        make_images_labels(make=make, dataset=dataset, image_dir_label=image_dir_label,
                           train_file=TRAIN_TXT_FILE, test_file=TEST_TXT_FILE)


if __name__ == "__main__":
    main()
