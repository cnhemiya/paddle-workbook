#!/usr/bin/bash

# 一键获取数据

# 获取数据
bash get-data.sh train.zip data33408

# 生成数据集列表
paddlex --split_dataset --format ImageNet --dataset_dir ./dataset/train --val_value 0.2 --test_value 0.1

# 检查数据
bash check-data.sh train
