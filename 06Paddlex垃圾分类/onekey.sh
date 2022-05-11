#!/usr/bin/bash

# 一键获取数据

zip_file="train.zip"
ais_dir="data33408"
sub_data_dir="train"
data_dir="./dataset/$sub_data_dir"

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

labels_file="$data_dir/labels.txt"

# 获取数据
if [ ! -d "$data_dir" ]; then
    bash get-data.sh "$zip_file" "$ais_dir"
fi

# 生成数据集列表
python3 make-dataset.py all $data_dir $dataset_list

echo "$labels_txt">"$labels_file"

# 检查数据
bash check-data.sh "$sub_data_dir"
