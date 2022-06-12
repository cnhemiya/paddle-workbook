#!/usr/bin/bash

# 图像分割，训练，量化，一键任务脚本

# 模型名称
MODEL="HRNet"
# HRNET_WIDTH 可选 18,48，高分辨率分支中特征层的通道数量
HRNET_WIDTH="18"
# 数据集目录
DATASET="./dataset/detroit_streetscape"
# 保存的目录
BASE_SAVE_DIR="./output/${MODEL}_${HRNET_WIDTH}"
# 导出模型的输入大小，默认 None，或者修改[n,c,w,h] --fixed_input_shape=[-1,3,512,512]
FIXED_INPUT_SHAPE="--fixed_input_shape=[-1,3,512,512]"

# 训练轮数
TRAIN_EPOCHS=96
# 训练单批次数量
TRAIN_BATCH_SIZE=1
# 训练学习率
TRAIN_LEARNING_RATE=0.01
# 训练保存间隔轮数
TRAIN_SAVE_INTERVAL_EPOCHS=1
# 训练预加载权重
TRAIN_PRETRAIN_WEIGHTS=""
# 训练模型保存的目录
TRAIN_SAVE_DIR="$BASE_SAVE_DIR/normal"
# 训练最佳模型保存的目录
TRAIN_BSET_SAVE_DIR="$TRAIN_SAVE_DIR/best_model"

# 量化训练轮数
QUANT_EPOCHS=64
# 量化训练单批次数量
QUANT_BATCH_SIZE=1
# 量化训练学习率
QUANT_LEARNING_RATE=0.001
# 量化训练保存间隔轮数
QUANT_SAVE_INTERVAL_EPOCHS=4
# 量化训练模型读取的目录
QUANT_MODEL_DIR="$TRAIN_BSET_SAVE_DIR"
# 量化训练模型保存的目录
QUANT_SAVE_DIR="$BASE_SAVE_DIR/quant"
# 量化训练最佳模型保存的目录
QUANT_BSET_SAVE_DIR="$QUANT_SAVE_DIR/best_model"

# 训练模型压缩文档
TRAIN_ZIP_FILE="${MODEL}_${HRNET_WIDTH}_${TRAIN_EPOCHS}e_${TRAIN_LEARNING_RATE}.tar.gz"
# 量化模型压缩文档
QUANT_ZIP_FILE="${MODEL}_${HRNET_WIDTH}_${QUANT_EPOCHS}e_${QUANT_LEARNING_RATE}_quant.tar.gz"

# 训练导出模型目录
TRAIN_INFER_SAVE_DIR="$BASE_SAVE_DIR/normal_infer"
# 量化导出模型目录
QUANT_INFER_SAVE_DIR="$BASE_SAVE_DIR/quant_infer"

# 训练导出模型压缩文档
TRAIN_INFER_ZIP_FILE="${MODEL}_${HRNET_WIDTH}_${TRAIN_EPOCHS}e_${TRAIN_LEARNING_RATE}_infer.tar.gz"
# 量化导出模型压缩文档
QUANT_INFER_ZIP_FILE="${MODEL}_${HRNET_WIDTH}_${QUANT_EPOCHS}e_${QUANT_LEARNING_RATE}_quant_infer.tar.gz"

echo "开始训练"
# 训练
python3 seq-train.py --dataset "$DATASET" \
    --epochs $TRAIN_EPOCHS \
    --batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $TRAIN_LEARNING_RATE \
    --model $MODEL \
    --hrnet_width $HRNET_WIDTH \
    --save_interval_epochs $TRAIN_SAVE_INTERVAL_EPOCHS \
    --pretrain_weights "$TRAIN_PRETRAIN_WEIGHTS" \
    --save_dir "$TRAIN_SAVE_DIR"

echo "保存并压缩训练模型"
tar -caf "$BASE_SAVE_DIR/$TRAIN_ZIP_FILE" "$TRAIN_BSET_SAVE_DIR"

echo "导出训练模型并压缩"
paddlex --export_inference --model_dir="$TRAIN_BSET_SAVE_DIR" --save_dir="$TRAIN_INFER_SAVE_DIR" $FIXED_INPUT_SHAPE
tar -caf "$BASE_SAVE_DIR/$TRAIN_INFER_ZIP_FILE" "$TRAIN_INFER_SAVE_DIR"

echo "开始量化"
# 量化
python3 seq-quant.py --dataset "$DATASET" \
    --epochs $QUANT_EPOCHS \
    --batch_size $QUANT_BATCH_SIZE \
    --learning_rate $QUANT_LEARNING_RATE \
    --save_interval_epochs $QUANT_SAVE_INTERVAL_EPOCHS \
    --model_dir "$QUANT_MODEL_DIR" \
    --save_dir "$QUANT_SAVE_DIR"

echo "保存并压缩量化模型"
tar -caf "$BASE_SAVE_DIR/$QUANT_ZIP_FILE" "$QUANT_BSET_SAVE_DIR"

echo "导出量化模型并压缩"
paddlex --export_inference --model_dir="$QUANT_BSET_SAVE_DIR" --save_dir="$QUANT_INFER_SAVE_DIR" $FIXED_INPUT_SHAPE
tar -caf "$BASE_SAVE_DIR/$QUANT_INFER_ZIP_FILE" "$QUANT_INFER_SAVE_DIR"

echo "结束任务"
