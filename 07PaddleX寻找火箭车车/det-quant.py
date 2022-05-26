#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-25 18:54
文档说明: 在线量化
"""


import paddlex as pdx
from paddlex import transforms as T
import mod.utils
import mod.args
import mod.config as config


def quant():
    # 解析命令行参数
    args = mod.args.QuantX()
    # 检查文件或目录是否存在
    args.check()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)

    # 定义训练和验证时的 transforms
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    train_transforms = T.Compose([
        T.MixupImage(mixup_epoch=-1),
        T.RandomDistort(),
        T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]),
        T.RandomCrop(),
        T.RandomHorizontalFlip(),
        T.BatchRandomResize(target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
                            interp='RANDOM'),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = T.Compose([
        T.Resize(target_size=608, interp='CUBIC'),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    # 量化器配置
    quant_config = {
        # weight预处理方法，默认为None，代表不进行预处理；当需要使用`PACT`方法时设置为`"PACT"`
        'weight_preprocess_type': None,

        # activation预处理方法，默认为None，代表不进行预处理`
        'activation_preprocess_type': None,

        # weight量化方法, 默认为'channel_wise_abs_max', 此外还支持'channel_wise_abs_max'
        'weight_quantize_type': 'channel_wise_abs_max',

        # activation量化方法, 默认为'moving_average_abs_max', 此外还支持'abs_max'
        'activation_quantize_type': 'moving_average_abs_max',

        # weight量化比特数, 默认为 8
        'weight_bits': 16,

        # activation量化比特数, 默认为 8
        'activation_bits': 16,

        # 'moving_average_abs_max'的滑动平均超参, 默认为0.9
        'moving_rate': 0.9,

        # 需要量化的算子类型
        'quantizable_layer_type': ['Conv2D', 'Linear']
    }

    # 加载模型
    print("读取模型 。。。读取路径：{}".format(args.model_dir))
    model = pdx.load_model(args.model_dir)

    # 模型训练
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/classification.md
    # 参数调整：https://gitee.com/paddlepaddle/PaddleX/blob/develop/docs/parameters.md
    # 可使用 VisualDL 查看训练指标，参考：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/visualdl.md
    print("开始训练 。。。保存路径：{}".format(args.save_dir))
    model.quant_aware_train(num_epochs=args.epochs,
                            train_dataset=train_dataset,
                            train_batch_size=args.batch_size,
                            eval_dataset=eval_dataset,
                            save_interval_epochs=args.save_interval_epochs,
                            save_dir=args.save_dir,
                            learning_rate=args.learning_rate,
                            warmup_steps=args.warmup_steps,
                            warmup_start_lr=args.warmup_start_lr,
                            lr_decay_epochs=args.lr_decay_epochs,
                            lr_decay_gamma=args.lr_decay_gamma,
                            resume_checkpoint=args.resume_checkpoint,
                            quant_config=quant_config,
                            use_vdl=True)
    print("结束训练 。。。保存路径：{}".format(args.save_dir))


def main():
    # 在线量化
    quant()


if __name__ == '__main__':
    main()
