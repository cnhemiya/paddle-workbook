#!/usr/bin/bash

# 一键获取数据

# 获取数据
bash get-data.sh

# 生成训练集和测试集图像路径和标签的文本文件
python3 make-images-labels.py ./dataset rps-cv-images/rock 0 rps-cv-images/scissors 1 rps-cv-images/paper 2

# 检查数据
bash check-data.sh
