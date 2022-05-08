#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-08 21:52
文档说明: 训练
"""


import paddle
import paddle.metric
import paddle.callbacks
import mod.utils
import mod.config as config


def train(net):
    # 打印分类信息
    config.print_num_classes()
    # 解析命令行参数
    args = config.train_args()
    # 使用transform对数据集做归一化
    transform = config.transform()
    # 训练数据集
    train_dataset = config.train_dataset(transform)
    # 测试数据集
    test_dataset = config.test_dataset(transform)
    # net 转为 paddle.Model 模型
    model = paddle.Model(net)
    # 优化器  SGD Adam Adadelta Adagrad
    optim = paddle.optimizer.Adam(
        learning_rate=args.learning_rate, parameters=model.parameters())
    # 配置模型
    model.prepare(optimizer=optim, loss=paddle.nn.CrossEntropyLoss(),
                  metrics=paddle.metric.Accuracy())
    # 读取模型参数
    if args.load_dir != "":
        config.load_model(model=model, loda_dir=args.load_dir,
                          reset_optimizer=False)    
    # 时间 ID
    time_id = mod.utils.time_id()
    # 输出 VisualDL 日志
    callback = None
    if (args.log):
        callback = paddle.callbacks.VisualDL(log_dir=config.get_log_dir(time_id=time_id))
    # 训练模型
    model.fit(train_data=train_dataset, epochs=args.epochs,
              batch_size=args.batch_size, num_workers=args.num_workers, verbose=1, callbacks=callback)
    # 评估模型
    result = model.evaluate(eval_data=test_dataset,
                            batch_size=args.batch_size, num_workers=args.num_workers, verbose=1)
    # 打印结果，loss 平均损失, acc 准确率
    print(result)
    # 保存模型参数和模型结果
    if not args.no_save:
        save_path = config.save_model(model, time_id=time_id)
        config.save_report(save_path=save_path, id=time_id,
                           args=args, eval_result=result)


def main():
    # 解析命令行参数
    args = config.train_args()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)
    # 网络模型
    net = config.net()
    # 网络模型信息
    if (args.summary):
        params_info = paddle.summary(
            net, (1, config.IMAGE_C, config.IMAGE_H, config.IMAGE_W))
        print(params_info)
    else:
        # 训练
        train(net)
    # config.user_cude(False)
    # net = config.net()
    # params_info = paddle.summary(
    # net, (1, config.IMAGE_C, config.IMAGE_H, config.IMAGE_W))
    # print(params_info)


if __name__ == '__main__':
    main()
