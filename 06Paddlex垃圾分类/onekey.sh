#!/usr/bin/bash

# 一键获取数据

# 获取数据
bash get-data.sh

# 生成训练集测试集列表
python3 make-dataset.py all ./dataset train/0 0 \
        train/1 1 train/2 2 train/3 3 train/4 4 train/5 5 \
        train/6 6 train/7 7 train/8 8 train/9 9 train/10 10 \
        train/11 11 train/12 12 train/13 13 train/14 14 train/15 15 \
        train/16 16 train/17 17 train/18 18 train/19 19 train/20 20 \
        train/21 21 train/22 22 train/23 23 train/24 24 train/25 25 \
        train/26 26 train/27 27 train/28 28 train/29 29 train/30 30 \
        train/31 31 train/32 32 train/33 33 train/34 34 train/35 35 \
        train/36 36 train/37 37 train/38 38 train/39 39 \

# 检查数据
bash check-data.sh
