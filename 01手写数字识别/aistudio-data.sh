#!/usr/bin/bash

from_dir=/home/aistudio/data/data65
to_dir=/home/aistudio/dataset

gzip -dfq $from_dir/train-images-idx3-ubyte.gz
gzip -dfq $from_dir/train-labels-idx1-ubyte.gz
gzip -dfq $from_dir/t10k-images-idx3-ubyte.gz
gzip -dfq $from_dir/t10k-labels-idx1-ubyte.gz

cp $from_dir/train-images-idx3-ubyte $to_dir/train-images.idx3-ubyte
cp $from_dir/train-labels-idx1-ubyte $to_dir/train-labels.idx1-ubyte
cp $from_dir/t10k-images-idx3-ubyte $to_dir/t10k-images.idx3-ubyte
cp $from_dir/t10k-labels-idx1-ubyte $to_dir/t10k-labels.idx1-ubyte

echo "复制数据完毕。。。"
