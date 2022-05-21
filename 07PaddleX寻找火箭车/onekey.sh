#!/usr/bin/bash

# 一键获取数据

# 数据压缩包
zip_file="train.zip"
# aistudio 数据目录
ais_dir="data33408"
# 解压后的数据目录
sub_data_dir="train"
# 数据目录
data_dir="./dataset/$sub_data_dir"

# 分类标签
labels_txt="0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39"

# 子目录对应的分类标签
dataset_list="0 0
1 1
2 2
3 3
4 4
5 5
6 6
7 7
8 8
9 9
10 10
11 11
12 12
13 13
14 14
15 15
16 16
17 17
18 18
19 19
20 20
21 21
22 22
23 23
24 24
25 25
26 26
27 27
28 28
29 29
30 30
31 31
32 32
33 33
34 34
35 35
36 36
37 37
38 38
39 39"

# 分类标签文件
labels_file="$data_dir/labels.txt"

# 获取数据
if [ ! -d "$data_dir" ]; then
    bash get-data.sh "$zip_file" "$ais_dir"
fi

# 生成数据集列表
python3 make-dataset.py all $data_dir $dataset_list

# 生成分类标签
echo "$labels_txt">"$labels_file"

# 检查数据
bash check-data.sh "$sub_data_dir"
