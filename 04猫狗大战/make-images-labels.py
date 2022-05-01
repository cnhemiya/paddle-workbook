#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-19 02:04
"""


import os
import sys


TRAIN_TXT_FILE = "train-images-labels.txt"
TEST_TXT_FILE = "test-images-labels.txt"


HELP_DOC = """
使用示例：

make-images-labels.py ./dataset img_dir1 0 img_dir2 1 img_dir3 2

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
    dataset = args[1]
    args = args[2:]
    image_dir_label = []
    for i in range(0, len(args), 2):
        idl = []
        idl.append(args[i])
        idl.append(args[i+1])
        image_dir_label.append(idl)
    return dataset, image_dir_label


def make_images_labels(dataset: str, image_dir_label, train_file: str, test_file: str):
    train_data = []
    test_data = []
    for i in image_dir_label:
        data = images_lables_data(
            dataset_dir=dataset, image_dir=i[0], label=int(i[1]))
        train_size = int(len(data) * 0.8)
        train_data.extend(data[:train_size])
        test_data.extend(data[train_size+1:])
    write_file(os.path.join(dataset, train_file), train_data)
    write_file(os.path.join(dataset, test_file), test_data)


def main():
    if (len(sys.argv) < 4):
        print(HELP_DOC)
    else:
        dataset, image_dir_label = arg_parse()
        make_images_labels(dataset=dataset, image_dir_label=image_dir_label,
                           train_file=TRAIN_TXT_FILE, test_file=TEST_TXT_FILE)


if __name__ == "__main__":
    main()
