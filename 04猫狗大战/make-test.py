#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-01 18:44
"""


import os


TRAIN_TXT_FILE = "train-images-labels.txt"
TEST_TXT_FILE = "test-images-labels.txt"


DATASET_DIR = "./dataset"
TEST_DIR = "catVSdog/test"
TEST_LABEL = DATASET_DIR + "/catVSdog/submit_example.csv"


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


def make_images_labels(dataset: str, test_label: str, test_file: str):
    lines = []
    with open(test_label, "r") as f:
        lines = f.readlines()
    lines = lines[1:]
    test_data = []
    lable = {"cat": 0, "dog": 1}
    for i in lines:
        i = i.strip("\n")
        data = i.split(",")
        img = os.path.join(TEST_DIR, data[0] + ".jpg")
        id = lable[data[1]]
        test_data.append("{} {}\n".format(img, id))
    write_file(os.path.join(dataset, test_file), test_data)


def main():
    make_images_labels(dataset=DATASET_DIR,
                       test_label=TEST_LABEL, test_file=TEST_TXT_FILE)
    print("生成测试集: {}".format(os.path.join(DATASET_DIR, TEST_TXT_FILE)))

if __name__ == "__main__":
    main()
