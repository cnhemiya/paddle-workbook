#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-08 21:52
文档说明: 测试
"""


import paddle
import paddle.metric
import mod.config as config


def test(net):
    # 解析命令行参数
    args = config.test_args()
    # 使用transform对数据集做归一化
    transform = config.transform()
    # 测试数据集
    test_dataset = config.test_dataset(transform)
    # net 转为 paddle.Model 模型
    model = paddle.Model(net)
    # 配置模型
    model.prepare(loss=paddle.nn.CrossEntropyLoss(),
                  metrics=paddle.metric.Accuracy())
    # 读取模型参数
    config.load_model(model=model, loda_dir=args.load_dir,
                      reset_optimizer=True)
    # 评估模型
    result = model.evaluate(eval_data=test_dataset,
                            batch_size=args.batch_size, num_workers=args.num_workers, verbose=1)
    # 打印结果，loss 平均损失, acc 准确率
    print(result)


def main():
    # 解析命令行参数
    args = config.test_args()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)
    # 网络模型
    net = config.net()
    # 测试
    test(net)


if __name__ == '__main__':
    main()
