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


def train():
    # 解析命令行参数
    args = config.train_args()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)

    # 定义训练和验证时的 transforms
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    train_transforms = T.Compose([
        T.RandomCrop(crop_size=224),
        T.RandomHorizontalFlip(),
        T.Normalize()])

    eval_transforms = T.Compose([
        T.ResizeByShort(short_size=256),
        T.CenterCrop(crop_size=224),
        T.Normalize()
    ])

    # 定义训练和验证所用的数据集
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
    train_dataset = pdx.datasets.ImageNet(
        data_dir=config.DATASET_PATH,
        file_list=config.TRAIN_LIST_PATH,
        label_list=config.LABEL_LIST_PATH,
        transforms=train_transforms,
        shuffle=True)

    eval_dataset = pdx.datasets.ImageNet(
        data_dir=config.DATASET_PATH,
        file_list=config.TEST_LIST_PATH,
        label_list=config.LABEL_LIST_PATH,
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
    # 恢复训练时指定上次训练保存的模型路径
    resume_dir = None

    # 加载模型权重
    if (args.weights != ""):
        pretrain_weights = os.path.join(args.weights, "model.pdparams")

    # 恢复训练
    if (args.resume != ""):
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
                learning_rate=args.learning_rate,
                lr_decay_epochs=[4, 6, 8],
                save_dir=save_dir,
                pretrain_weights=pretrain_weights,
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
