#!/usr/bin/bash

# 一键获取数据

# 数据压缩包
zip_file="detroit_streetscape.zip"
# aistudio 数据目录
ais_dir="data152045"
# 解压后的数据目录
sub_data_dir="detroit_streetscape"
# 数据目录
data_dir="./dataset/$sub_data_dir"

# 分类标签
labels_txt="car"

# 子目录对应的分类标签
dataset_list=""

# 分类标签文件
labels_file="$data_dir/labels.txt"

# 获取数据
if [ ! -d "$data_dir" ]; then
    bash get-data.sh "$zip_file" "$ais_dir"
fi

# 生成数据集列表
# python3 make-dataset.py all $data_dir $dataset_list
# paddlex --split_dataset --format VOC --dataset_dir "$data_dir" --val_value 0.2 --test_value 0.1

# 生成分类标签
# echo "$labels_txt">"$labels_file"

# 检查数据
bash check-data.sh "$sub_data_dir"
