#!/usr/bin/bash

# 检查 dataset 目录下的数据是否存在

dataset_dir="./dataset"
data_files="
$dataset_dir/train_list.txt
$dataset_dir/eval_list.txt
$dataset_dir/test_list.txt
$dataset_dir/labels.txt
"
data_dirs="
./dataset
./output
$dataset_dir/train/0
$dataset_dir/train/1
$dataset_dir/train/2
$dataset_dir/train/3
$dataset_dir/train/4
$dataset_dir/train/5
$dataset_dir/train/6
$dataset_dir/train/7
$dataset_dir/train/8
$dataset_dir/train/9
$dataset_dir/train/10
$dataset_dir/train/11
$dataset_dir/train/12
$dataset_dir/train/13
$dataset_dir/train/14
$dataset_dir/train/15
$dataset_dir/train/16
$dataset_dir/train/17
$dataset_dir/train/18
$dataset_dir/train/19
$dataset_dir/train/20
$dataset_dir/train/21
$dataset_dir/train/22
$dataset_dir/train/23
$dataset_dir/train/24
$dataset_dir/train/25
$dataset_dir/train/26
$dataset_dir/train/27
$dataset_dir/train/28
$dataset_dir/train/29
$dataset_dir/train/30
$dataset_dir/train/31
$dataset_dir/train/32
$dataset_dir/train/33
$dataset_dir/train/34
$dataset_dir/train/35
$dataset_dir/train/36
$dataset_dir/train/37
$dataset_dir/train/38
$dataset_dir/train/39
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
