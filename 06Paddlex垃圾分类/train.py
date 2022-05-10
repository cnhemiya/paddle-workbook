#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-09 19:16
文档说明: 训练
"""


import os
import paddlex as pdx
from paddlex import transforms as T
import mod.utils
import mod.config as config


# 训练 transforms 图像大小
TRAIN_IMAGE_SIZE = 224

# 评估 transforms 图像大小
EVAL_IMAGE_SIZE = 256

# 测试 transforms 图像大小
TEST_IMAGE_SIZE = 224


def train():
    # 解析命令行参数
    args = config.train_args()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)

    # 数据集目录和文件路径
    dataset_path = config.DATASET_PATH
    if (args.dataset != ""):
        dataset_path = os.path.join(config.DATASET_PATH, args.dataset)
    train_list_path = os.path.join(dataset_path, config.TRAIN_LIST_PATH)
    eval_list_path = os.path.join(dataset_path, config.EVAL_LIST_PATH)
    label_list_path = os.path.join(dataset_path, config.LABEL_LIST_PATH)
    mod.utils.check_path(dataset_path)
    mod.utils.check_path(train_list_path)
    mod.utils.check_path(eval_list_path)
    mod.utils.check_path(label_list_path)

    # 定义训练和验证时的 transforms
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    train_transforms = T.Compose([
        T.RandomCrop(crop_size=TRAIN_IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.Normalize()])

    eval_transforms = T.Compose([
        T.ResizeByShort(short_size=EVAL_IMAGE_SIZE),
        T.CenterCrop(crop_size=TRAIN_IMAGE_SIZE),
        T.Normalize()])

    # 定义训练和验证所用的数据集
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
    train_dataset = pdx.datasets.ImageNet(
        data_dir=dataset_path,
        file_list=train_list_path,
        label_list=label_list_path,
        transforms=train_transforms,
        shuffle=True)

    eval_dataset = pdx.datasets.ImageNet(
        data_dir=dataset_path,
        file_list=eval_list_path,
        label_list=label_list_path,
        transforms=eval_transforms)

    # 分类数量
    num_classes = len(train_dataset.labels)
    # 获取 PaddleX 模型
    model, model_name = config.pdx_cls_model(
        model_name=args.model, num_classes=num_classes)

    # 时间 ID
    time_id = mod.utils.time_id()
    # 输出保存的目录
    save_dir = config.get_save_dir(time_id=time_id)

    # 模型权重
    pretrain_weights = "IMAGENET"
    # 加载模型权重
    if (args.weights != ""):
        mod.utils.check_path(args.weights)
        pretrain_weights = args.weights

    # 恢复训练时指定上次训练保存的模型路径
    resume_dir = None
    # 恢复训练
    if (args.resume != ""):
        mod.utils.check_path(args.resume)
        pretrain_weights = None
        resume_dir = args.resume

    print("开始训练。。。")

    # 模型训练
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/classification.md
    # 参数调整：https://gitee.com/paddlepaddle/PaddleX/blob/develop/docs/parameters.md
    # 可使用 VisualDL 查看训练指标，参考：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/visualdl.md
    model.train(num_epochs=args.epochs,
                train_dataset=train_dataset,
                train_batch_size=args.batch_size,
                eval_dataset=eval_dataset,
                save_interval_epochs=args.save_interval_epochs,
                save_dir=save_dir,
                pretrain_weights=pretrain_weights,
                learning_rate=args.learning_rate,
                lr_decay_epochs=mod.utils.str_to_list(args.lr_decay_epochs),
                lr_decay_gamma=args.lr_decay_gamma,
                resume_checkpoint=resume_dir,
                use_vdl=True)

    # 保存报表
    config.save_report_x(save_dir=save_dir, id=time_id,
                         model=model_name, args=args)

    print("结束训练。。。")


def main():
    # 解析命令行参数
    args = config.train_args()
    # PaddleX 模型名称
    if (args.model_list):
        model_list = config.pdx_cls_model_name()
        print(model_list)
    else:
        # 训练
        train()


if __name__ == '__main__':
    main()
