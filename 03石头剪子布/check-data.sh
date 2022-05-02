#!/usr/bin/bash

# 检查 dataset 目录下的数据是否存在

dataset_dir="./dataset"
data_files="
$dataset_dir/train-images-labels.txt
$dataset_dir/test-images-labels.txt
"
data_dirs="
$dataset_dir/rps-cv-images/rock
$dataset_dir/rps-cv-images/scissors
$dataset_dir/rps-cv-images/paper
./dataset
./params
./log
"

# 检查文件
check_files() {
    paths=$@
    for i in $paths; do
        if [ -f "$i" ]; then
            echo "检查文件: $i  --  存在"
        else
            echo "检查文件: $i  --  不存在"
        fi
    done
}

# 检查文件夹
check_dirs() {
    paths=$@
    for i in $paths; do
        if [ -d "$i" ]; then
            echo "检查文件夹: $i  --  存在"
        else
            echo "检查文件夹: $i  --  不存在"
        fi
    done
}

check_files ${data_files[@]}
check_dirs ${data_dirs[@]}
