#!/usr/bin/bash

# 获取数据到 dataset 目录下

dir_list="./dataset ./output"
dataset_dir="./dataset"
ais_dataset_dir="../data/data33408"
zip_files="train.zip"
# data_files="train-images-labels.txt test-images-labels.txt"
local_files="labels.txt"

# 创建文件夹
make_dir_list() {
    for i in $dir_list; do
        if [ ! -d "$i" ]; then
            mkdir "$i"
            touch "$i/.keep"
        fi
    done
}

# 解压缩文件
unzip_file() {
    file="$1"
    dir="$2"
    ext="${file##*.}"
    if [ -f "$file" ]; then
        echo "解压文件: $file"
        if [ $ext == "zip" ]; then
            unzip -oq "$file" -d "$dir"
        elif [ $ext == "gz" ]; then
            gzip -dqkfN "$file"
        fi
    fi
}

# 获取数据
get_data() {
    file="$1"
    if [ -f "$dataset_dir/$file" ]; then
        echo "找到文件: $dataset_dir/$file"
    elif [ -f "$ais_dataset_dir/$file" ]; then
        echo "找到文件: $ais_dataset_dir/$file"
        echo "复制文件到: $dataset_dir/$file"
        cp "$ais_dataset_dir/$file" "$dataset_dir/$file"
    fi
    if [ "$2" == "zip" ]; then
        unzip_file "$dataset_dir/$file" "$dataset_dir"
    fi
}

# 获取本地数据
get_local_data() {
    file="$1"
    if [ -f "$dataset_dir/$file" ]; then
        echo "找到文件: $dataset_dir/$file"
    elif [ -f "$file" ]; then
        echo "找到文件: $file"
        echo "复制文件到: $dataset_dir/$file"
        cp "$file" "$dataset_dir/$file"
    fi
}

# 获取全部压缩文件数据
get_all_zip_data() {
    for i in $zip_files; do
        get_data "$i" "zip"
    done
}

# 获取全部文件数据
get_all_file_data() {
    for i in $data_files; do
        get_data "$i"
    done
}

# 获取全部本地文件数据
get_all_local_data() {
    for i in $local_files; do
        get_local_data "$i"
    done
}

make_dir_list
get_all_zip_data
# get_all_file_data
get_all_local_data
