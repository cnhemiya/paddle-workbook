# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-08 21:52
文档说明: 配置
"""


import paddle
import paddle.vision.transforms as pptf
import argparse
import mod.dataset
import mod.lenet
import mod.utils
import mod.report

# 数据集路径
DATA_DIR_PATH = "./dataset/"
# 训练数据
TRAIN_DATA_PATH = DATA_DIR_PATH + "train-images.idx3-ubyte"
# 训练标签
TRAIN_LABEL_PATH = DATA_DIR_PATH + "train-labels.idx1-ubyte"
# 测试数据
TEST_DATA_PATH = DATA_DIR_PATH + "t10k-images.idx3-ubyte"
# 测试标签
TEST_LABLE_PATH = DATA_DIR_PATH + "t10k-labels.idx1-ubyte"

# 模型参数保存的文件夹
SAVE_DIR = "./params/"
# 最佳参数保存的文件夹
SAVE_BAST_DIR = "base"
# 模型参数保存的前缀
SAVE_PREFIX = "model"

# report 文件名
REPORT_FILE = "report.json"


def user_cude(cuda=True):
    """
    使用 cuda gpu 还是 cpu 运算

    Args:
        cuda (bool, optional): cuda, 默认 True.
    """
    paddle.device.set_device(
        "gpu:0") if cuda else paddle.device.set_device("cpu")


def transform():
    """
    获取 transform 对数据进行转换

    Returns:
        Compose: 转换数据的操作组合
    """
    return pptf.Compose([pptf.Normalize(mean=[127.5], std=[127.5], data_format='CHW')])


def train_dataset(transform: pptf.Compose):
    """
    获取训练数据集

    Args:
        transform (Compose): 转换数据的操作组合

    Returns:
        MNIST: MNIST 手写数据集解析
    """
    return mod.dataset.MNIST(image_path=TRAIN_DATA_PATH, label_path=TRAIN_LABEL_PATH, transform=transform)


def test_dataset(transform):
    """
    获取测试数据集

    Args:
        transform (Compose): 转换数据的操作组合

    Returns:
        MNIST: MNIST 手写数据集解析
    """
    return mod.dataset.MNIST(image_path=TEST_DATA_PATH, label_path=TEST_LABLE_PATH, transform=transform)


def net(num_classes=10):
    """
    获取网络模型

    Args:
        num_classes (int, optional): 分类数量, 默认 10.

    Returns:
        LeNet: LeNet 网络模型
    """
    return mod.lenet.LeNet(num_classes=num_classes)


def save_model(model, save_dir=SAVE_DIR, save_prefix=SAVE_PREFIX):
    """
    保存模型参数

    Args:
        model (paddle.Model): 网络模型
        save_dir (str, optional): 保存模型的文件夹, 默认 SAVE_DIR.
        save_prefix (str, optional): 保存模型的前缀, 默认 SAVE_PREFIX.

    Returns:
        save_path (str): 保存的路径
        time_str (str): 时间 id
    """
    time_str = mod.utils.time_str()
    save_path = save_dir + time_str
    model.save(save_path + "/" + save_prefix)
    return save_path, time_str


def load_model(model, loda_dir="", save_prefix=SAVE_PREFIX, reset_optimizer=False):
    """
    读取模型参数

    Args:
        model (paddle.Model): 网络模型
        loda_dir (str, optional): 读取模型的文件夹, 默认 "".
        save_prefix (str, optional): 保存模型的前缀, 默认 SAVE_PREFIX.
        reset_optimizer (bool, optional): 重置 optimizer 参数, 默认 False 不重置.
    """
    load_path = SAVE_DIR + loda_dir + "/"
    mod.utils.check_path(load_path)
    model.load(path=load_path + save_prefix, reset_optimizer=reset_optimizer)


def save_report(save_path: str, id: str, args=None, eval_result=None):
    """
    保存结果报表

    Args:
        save_path (str): 保存的路径
        id (str): 报表 id
        args (_type_, optional): 命令行参数, Defaults to None.
        eval_result (list, optional): 评估结果, 默认 None.

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
    report.save(save_path + "/" + REPORT_FILE)


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
    arg_parse.add_argument("--batch-size", type=int, default=128,
                           dest="batch_size", metavar="", help="一批次数量，默认 128")
    arg_parse.add_argument("--num-workers", type=int, default=2,
                           dest="num_workers", metavar="", help="线程数量，默认 2")
    arg_parse.add_argument("--no-save", action="store_true",
                           dest="no_save", help="是否保存模型参数，默认保存, 选择后不保存模型参数")
    arg_parse.add_argument("--load-dir", dest="load_dir", default="",
                           metavar="", help="读取模型参数，读取 params 目录下的子文件夹, 默认不读取")
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
    arg_parse.add_argument("--batch-size", type=int, default=128,
                           dest="batch_size", metavar="", help="一批次数量，默认 128")
    arg_parse.add_argument("--num-workers", type=int, default=2,
                           dest="num_workers", metavar="", help="线程数量，默认 2")
    arg_parse.add_argument("--load-dir", dest="load_dir", default="best",
                           metavar="", help="读取模型参数，读取 params 目录下的子文件夹, 默认 best 目录")
    return arg_parse.parse_args()
