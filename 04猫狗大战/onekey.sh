#!/usr/bin/bash

# 一键获取数据

# 获取数据
bash get-data.sh

# 生成训练集图像路径和标签的文本文件
python3 make-images-labels.py train ./dataset catVSdog/train/cat 0 catVSdog/train/dog 1

# 生成测试集图像路径和标签的文本文件
python3 make-test.py

# 检查数据
bash check-data.sh
