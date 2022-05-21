#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-09 19:16
文档说明: 训练
"""


import paddlex as pdx
from paddlex import transforms as T
import mod.utils
import mod.config as config
import mod.args
import mod.pdx


# 训练 transforms 图像大小
TRAIN_IMAGE_SIZE = 224

# 评估 transforms 图像大小
EVAL_IMAGE_SIZE = 256

# 测试 transforms 图像大小
TEST_IMAGE_SIZE = 224


def train():
    # 解析命令行参数
    args = mod.args.TrainXDet()
    # 检查文件或目录是否存在
    args.check()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)

    # 定义训练和验证时的 transforms
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    train_transforms = T.Compose([
        T.MixupImage(mixup_epoch=-1), T.RandomDistort(),
        T.RandomExpand(im_padding_value=[
                       123.675, 116.28, 103.53]), T.RandomCrop(),
        T.RandomHorizontalFlip(), T.BatchRandomResize(
            target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
            interp='RANDOM'), T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = T.Compose([
        T.Resize(
            target_size=608, interp='CUBIC'), T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 定义训练和验证所用的数据集
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
    train_dataset = pdx.datasets.VOCDetection(
        data_dir=args.dataset,
        file_list=args.train_list,
        label_list=args.label_list,
        transforms=train_transforms,
        num_workers=args.num_workers,
        shuffle=True)

    eval_dataset = pdx.datasets.VOCDetection(
        data_dir=args.dataset,
        file_list=args.eval_list,
        label_list=args.label_list,
        transforms=eval_transforms,
        num_workers=args.num_workers,
        shuffle=False)

    # 分类数量
    num_classes = len(train_dataset.labels)
    # 获取 PaddleX 模型
    model, model_name = mod.pdx.pdx_det_model(
        model_name=args.model, backbone=args.backbone, num_classes=num_classes)

    print("开始训练 。。。模型：{}".format(model_name))

    # 模型训练
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/classification.md
    # 参数调整：https://gitee.com/paddlepaddle/PaddleX/blob/develop/docs/parameters.md
    # 可使用 VisualDL 查看训练指标，参考：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/visualdl.md
    model.train(num_epochs=args.epochs,
                train_dataset=train_dataset,
                train_batch_size=args.batch_size,
                eval_dataset=eval_dataset,
                save_interval_epochs=args.save_interval_epochs,
                save_dir=args.save_dir,
                pretrain_weights=args.pretrain_weights,
                learning_rate=args.learning_rate,
                lr_decay_epochs=args.lr_decay_epochs,
                lr_decay_gamma=args.lr_decay_gamma,
                resume_checkpoint=args.resume_checkpoint,
                use_vdl=True)

    print("结束训练 。。。模型：{}".format(model_name))


def main():
    # 解析命令行参数
    args = mod.args.TrainXDet()
    # PaddleX 模型名称
    if (args.model_list):
        mod.pdx.print_pdx_det_model_name()
    else:
        # 训练
        train()


if __name__ == '__main__':
    main()
