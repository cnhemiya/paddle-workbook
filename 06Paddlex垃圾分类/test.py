#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import paddlex as pdx
from paddlex import transforms as T
import mod.config as config
import mod.utils


# 训练 transforms 图像大小
TRAIN_IMAGE_SIZE = 224

# 评估 transforms 图像大小
EVAL_IMAGE_SIZE = 256

# 测试 transforms 图像大小
TEST_IMAGE_SIZE = 256


def main():
    # 解析命令行参数
    args = config.test_args()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)

    # 数据集目录和文件路径
    dataset_path = config.DATASET_PATH
    if (args.dataset != ""):
        dataset_path = os.path.join(config.DATASET_PATH, args.dataset)
    test_list_path = os.path.join(dataset_path, config.TEST_LIST_PATH)
    mod.utils.check_path(dataset_path)
    mod.utils.check_path(test_list_path)

    # 定义训练和验证时的 transforms
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    test_transforms = T.Compose([
        T.ResizeByShort(short_size=TEST_IMAGE_SIZE),
        T.CenterCrop(crop_size=TRAIN_IMAGE_SIZE),
        T.Normalize()])

    # 数据集解析
    image_paths, labels = mod.utils.parse_dataset(
        dataset_path, dataset_list_path=test_list_path, shuffle=True)

    # 模型文件目录
    model_path = args.model
    mod.utils.check_path(model_path)
    # 读取模型
    model = pdx.load_model(model_path)

    # 样本数量
    sample_num = len(labels)
    # 测试几轮
    test_epochs = args.epochs

    for i in range(test_epochs):
        # 正确数量
        ok_num = 0
        # 错误数量
        err_num = 0
        print("开始测试 。。。第 {} 轮".format(i + 1))
        # 分类模型预测接口
        result = model.predict(img_file=image_paths,
                               transforms=test_transforms)
        # 计算结果
        for i in range(len(result)):
            data = result[i]
            data = data[0]
            if data["category_id"] == labels[i]:
                ok_num += 1
            else:
                err_num += 1
        print("样本数量: {},  正确率: {:<.6f},  正确样本: {},  错误样本: {}".format(
            sample_num, ok_num/sample_num, ok_num, err_num))
    print("结束测试 。。。")


if __name__ == '__main__':
    main()
