# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-16 11:37
文档说明: 配置
"""


import os
import argparse
import paddle
import paddle.vision.transforms as pptf
import mod.dataset
import mod.utils
import mod.report
import mod.alexnet


# 分类文本, 按照标签排列
CLASS_TXT = ["非眼疾", "眼疾"]

# 图像通道 彩色 3, 灰度 1
IMAGE_C = 3
# 图像高
IMAGE_H = 224
# 图像宽
IMAGE_W = 224

# 数据集路径
DATASET_PATH = "./dataset/"
# 训练数据
TRAIN_DATA_PATH = DATASET_PATH + "train-images-labels.txt"
# 测试数据
TEST_DATA_PATH = DATASET_PATH + "test-images-labels.txt"

# 模型参数保存的文件夹
SAVE_DIR = "./params/"
# 最佳参数保存的文件夹
SAVE_BAST_DIR = "best"
# 最佳参数保存的路径
SAVE_BEST_PATH = SAVE_DIR + SAVE_BAST_DIR + "/"
# 模型参数保存的前缀
SAVE_PREFIX = "model"

# 日志保存的路径
LOG_DIR = "./log"

# report 文件名
REPORT_FILE = "report.json"


def net(num_classes=2, pool_kernel_size=2, conv1_paddling=2, fc1_in_features=9216):
    """
    获取网络模型

    Args:
        num_classes (int, optional): 分类数量, 默认 2
        pool_kernel_size (int, optional): 池化层核大小, 默认 2
        conv1_paddling (int, optional): 第一层卷积层填充, 默认 2,
            输入图像大小为 224 x 224 填充 2
        fc1_in_features (int, optional): 第一层全连接层输入特征数量, 默认 9216, 

    Returns:
        AlexNet: AlexNet 网络模型
    """
    return mod.alexnet.AlexNet(num_classes=num_classes, pool_kernel_size=pool_kernel_size,
                               conv1_paddling=conv1_paddling, fc1_in_features=fc1_in_features)


def transform():
    """
    获取 transform 对数据进行转换

    Returns:
        Compose: 转换数据的操作组合
    """
    # Resize: 调整图像大小, Normalize: 图像归一化处理
    return pptf.Compose([pptf.Resize(size=[IMAGE_H, IMAGE_W]), pptf.Normalize(mean=[127.5, 127.5, 127.5],
                                                                              std=[127.5, 127.5, 127.5], data_format='HWC')])


def image_to_tensor(image):
    """
    图像数据转 tensor

    Returns:
        tensor: 转换后的 tensor 数据
    """
    # 图像数据格式 CHW
    data = image.reshape([1, IMAGE_C, IMAGE_H, IMAGE_W]).astype("float32")
    return paddle.to_tensor(data)


def train_dataset(transform: pptf.Compose):
    """
    获取训练数据集

    Args:
        transform (Compose): 转换数据的操作组合

    Returns:
        ImageClass: ImageClass 图像分类数据集解析
    """
    return mod.dataset.ImageClass(dataset_path=DATASET_PATH, images_labels_txt_path=TRAIN_DATA_PATH, transform=transform)


def test_dataset(transform):
    """
    获取测试数据集

    Args:
        transform (Compose): 转换数据的操作组合

    Returns:
        ImageClass: ImageClass 图像分类数据集解析
    """
    return mod.dataset.ImageClass(dataset_path=DATASET_PATH, images_labels_txt_path=TEST_DATA_PATH, transform=transform)


def get_log_dir(log_dir=LOG_DIR, time_id=mod.utils.time_id()):
    """
    获取 VisualDL 日志文件夹

    Args:
        log_dir (str, optional): 日志文件夹, 默认 LOG_DIR
        time_id (str, optional): 根据时间生成的字符串 ID

    Returns:
        str : VisualDL 日志文件夹
    """
    return os.path.join(log_dir, time_id)


def save_model(model, save_dir=SAVE_DIR, time_id=mod.utils.time_id(), save_prefix=SAVE_PREFIX):
    """
    保存模型参数

    Args:
        model (paddle.Model): 网络模型
        save_dir (str, optional): 保存模型的文件夹, 默认 SAVE_DIR
        time_id (str): 根据时间生成的字符串 ID
        save_prefix (str, optional): 保存模型的前缀, 默认 SAVE_PREFIX

    Returns:
        save_path (str): 保存的路径
    """
    save_path = os.path.join(save_dir, time_id)
    model.save(os.path.join(save_path, save_prefix))
    return save_path


def load_model(model, loda_dir="", save_prefix=SAVE_PREFIX, reset_optimizer=False):
    """
    读取模型参数

    Args:
        model (paddle.Model): 网络模型
        loda_dir (str, optional): 读取模型的文件夹, 默认 ""
        save_prefix (str, optional): 保存模型的前缀, 默认 SAVE_PREFIX
        reset_optimizer (bool, optional): 重置 optimizer 参数, 默认 False 不重置
    """
    load_path = os.path.join(SAVE_DIR, loda_dir)
    mod.utils.check_path(load_path)
    load_path = os.path.join(load_path, save_prefix)
    model.load(path=load_path, reset_optimizer=reset_optimizer)


def save_report(save_path: str, id: str, args=None, eval_result=None):
    """
    保存结果报表

    Args:
        save_path (str): 保存的路径
        id (str): 报表 id
        args (_type_, optional): 命令行参数, Defaults to None
        eval_result (list, optional): 评估结果, 默认 None

    Raises:
        Exception: eval_result 不能为 None
    """
    if eval_result == None:
        raise Exception("评估结果不能为 None")
    report = mod.report.Report()
    report.id = id
    report.loss = float(eval_result["loss"][0])
    report.acc = float(eval_result["acc"])
    report.epochs = args.epochs
    report.batch_size = args.batch_size
    report.learning_rate = float(args.learning_rate)
    report.save(os.path.join(save_path, REPORT_FILE))


def train_args():
    """
    返回训练命令行参数

    Returns:
        argparse: 命令行参数
    """
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--cpu", action="store_true",
                           dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
    arg_parse.add_argument("--learning-rate", type=float, default=0.001,
                           dest="learning_rate", metavar="", help="学习率，默认 0.001")
    arg_parse.add_argument("--epochs", type=int, default=2,
                           dest="epochs", metavar="", help="训练几轮，默认 2 轮")
    arg_parse.add_argument("--batch-size", type=int, default=2,
                           dest="batch_size", metavar="", help="一批次数量，默认 2")
    arg_parse.add_argument("--num-workers", type=int, default=2,
                           dest="num_workers", metavar="", help="线程数量，默认 2")
    arg_parse.add_argument("--no-save", action="store_true",
                           dest="no_save", help="是否保存模型参数，默认保存, 选择后不保存模型参数")
    arg_parse.add_argument("--load-dir", dest="load_dir", default="",
                           metavar="", help="读取模型参数，读取 params 目录下的子文件夹, 默认不读取")
    arg_parse.add_argument("--log", action="store_true",
                           dest="log", help="是否输出 VisualDL 日志，默认不输出")
    arg_parse.add_argument("--summary", action="store_true",
                           dest="summary", help="输出网络模型信息，默认不输出，选择后只输出信息，不会开启训练")
    return arg_parse.parse_args()


def test_args():
    """
    返回测试命令行参数

    Returns:
        argparse: 命令行参数
    """
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--cpu", action="store_true",
                           dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
    arg_parse.add_argument("--batch-size", type=int, default=2,
                           dest="batch_size", metavar="", help="一批次数量，默认 2")
    arg_parse.add_argument("--num-workers", type=int, default=2,
                           dest="num_workers", metavar="", help="线程数量，默认 2")
    arg_parse.add_argument("--load-dir", dest="load_dir", default="best",
                           metavar="", help="读取模型参数，读取 params 目录下的子文件夹, 默认 best 目录")
    return arg_parse.parse_args()


def user_cude(cuda=True):
    """
    使用 cuda gpu 还是 cpu 运算

    Args:
        cuda (bool, optional): cuda, 默认 True
    """
    paddle.device.set_device(
        "gpu:0") if cuda else paddle.device.set_device("cpu")
