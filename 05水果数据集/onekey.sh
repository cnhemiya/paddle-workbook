#!/usr/bin/bash

# 一键获取数据

# 获取数据
bash get-data.sh

# 生成训练集测试集图像路径和标签的文本文件
python3 make-images-labels.py all ./dataset fruits/apple 0 fruits/banana 1 fruits/grape 2 fruits/orange 3 fruits/pear 4

# 检查数据
bash check-data.sh
