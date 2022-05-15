# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-15 16:54
文档说明: 命令行参数解析
"""


import os
import argparse
import mod.utils
import mod.config as config


class Train():
    """
    返回训练命令行参数
    """

    def __init__(self, args=None) -> None:
        self.args = self.parse() if args == None else args
        self.cpu = args.cpu
        self.num_workers = args.num_workers
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.no_save = args.no_save
        self.load_dir = args.load_dir
        self.log = args.log
        self.summary = args.summary

    def parse(self):
        """
        返回命令行参数

        Returns:
            argparse: 命令行参数
        """
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("--cpu", action="store_true",
                               dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        arg_parse.add_argument("--num_workers", type=int, default=2,
                               dest="num_workers", metavar="", help="线程数量，默认 2")
        arg_parse.add_argument("--learning_rate", type=float, default=0.001,
                               dest="learning_rate", metavar="", help="学习率，默认 0.001")
        arg_parse.add_argument("--epochs", type=int, default=2,
                               dest="epochs", metavar="", help="训练几轮，默认 2 轮")
        arg_parse.add_argument("--batch_size", type=int, default=2,
                               dest="batch_size", metavar="", help="一批次数量，默认 2")
        arg_parse.add_argument("--num_workers", type=int, default=2,
                               dest="num_workers", metavar="", help="线程数量，默认 2")
        arg_parse.add_argument("--no_save", action="store_true",
                               dest="no_save", help="是否保存模型参数，默认保存, 选择后不保存模型参数")
        arg_parse.add_argument("--load_dir", dest="load_dir", default="",
                               metavar="", help="读取模型参数，读取 params 目录下的子文件夹, 默认不读取")
        arg_parse.add_argument("--log", action="store_true",
                               dest="log", help="是否输出 VisualDL 日志，默认不输出")
        arg_parse.add_argument("--summary", action="store_true",
                               dest="summary", help="输出网络模型信息，默认不输出，选择后只输出信息，不会开启训练")
        return arg_parse.parse_args()


class Test():
    """
    返回测试命令行参数
    """

    def __init__(self, args=None) -> None:
        self.args = self.parse() if args == None else args
        self.cpu = args.cpu
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.load_dir = args.load_dir

    def parse(self):
        """
        返回命令行参数

        Returns:
            argparse: 命令行参数
        """
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("--cpu", action="store_true",
                               dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        arg_parse.add_argument("--num_workers", type=int, default=2,
                               dest="num_workers", metavar="", help="线程数量，默认 2")
        arg_parse.add_argument("--batch_size", type=int, default=2,
                               dest="batch_size", metavar="", help="一批次数量，默认 2")
        arg_parse.add_argument("--load_dir", dest="load_dir", default="best",
                               metavar="", help="读取模型参数，读取 params 目录下的子文件夹, 默认 best 目录")
        return arg_parse.parse_args()


class Predict():
    """
    返回预测命令行参数
    """

    def __init__(self, args=None) -> None:
        self.args = self.parse() if args == None else args
        self.cpu = args.cpu
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.load_dir = args.load_dir
        self.no_save = args.no_save

    def parse(self):
        """
        返回命令行参数

        Returns:
            argparse: 命令行参数
        """
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("--cpu", action="store_true",
                               dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        arg_parse.add_argument("--num_workers", type=int, default=2,
                               dest="num_workers", metavar="", help="线程数量，默认 2")
        arg_parse.add_argument("--batch_size", type=int, default=1,
                               dest="batch_size", metavar="", help="一批次数量，默认 1")
        arg_parse.add_argument("--load_dir", dest="load_dir", default="best",
                               metavar="", help="读取模型参数，读取 params 目录下的子文件夹, 默认 best 目录")
        arg_parse.add_argument("--no_save", action="store_true",
                               dest="no_save", help="是否保存预测结果，默认保存, 选择后不保存预测结果")
        return arg_parse.parse_args()


class TrainX():
    """
    返回 PaddleX 训练命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        self.__dataset_path = dataset_path
        self.__train_list_path = train_list_path
        self.__eval_list_path = eval_list_path
        self.__label_list_path = label_list_path
        self.__save_dir_path = save_dir_path

        self.args = self.parse() if args == None else args
        self.cpu = self.args.cpu
        self.num_workers = "auto" if self.args.num_workers == 0 else self.args.num_workers

        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.learning_rate
        self.lr_decay_epochs = mod.utils.str_to_list(self.args.lr_decay_epochs)
        self.lr_decay_gamma = self.args.lr_decay_gamma
        self.save_interval_epochs = self.args.save_interval_epochs
        self.save_dir = save_dir_path if self.args.save_dir == "" else self.args.save_dir
        self.dataset = dataset_path if self.args.dataset == "" else os.path.join(
            dataset_path, self.args.dataset)
        self.model = self.args.model
        self.pretrain_weights = self.args.pretrain_weights
        self.resume_checkpoint = self.args.resume_checkpoint

        self.model_list = self.args.model_list
        self.train_list = os.path.join(self.dataset, self.args.train_list)
        self.eval_list = os.path.join(self.dataset, self.args.eval_list)
        self.label_list = os.path.join(self.dataset, self.args.label_list)

    def parse(self):
        """
        返回命令行参数

        Returns:
            argparse: 命令行参数
        """
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("--cpu", action="store_true",
                               dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        arg_parse.add_argument("--num_workers", type=int, default=0,
                               dest="num_workers", metavar="", help="线程数量，默认 auto，为CPU核数的一半")
        arg_parse.add_argument("--epochs", type=int, default=4,
                               dest="epochs", metavar="", help="训练几轮，默认 4 轮")
        arg_parse.add_argument("--batch_size", type=int, default=16,
                               dest="batch_size", metavar="", help="一批次数量，默认 16")
        arg_parse.add_argument("--learning_rate", type=float, default=0.025,
                               dest="learning_rate", metavar="", help="学习率，默认 0.025")
        arg_parse.add_argument("--lr_decay_epochs", dest="lr_decay_epochs", default="30 60 90",
                               metavar="", help="默认优化器的学习率衰减轮数。默认为 30 60 90")
        arg_parse.add_argument("--lr_decay_gamma", type=float, default=0.1,
                               dest="lr_decay_gamma", metavar="", help="默认优化器的学习率衰减率。默认为0.1")
        arg_parse.add_argument("--save_interval_epochs", type=int, default=1,
                               dest="save_interval_epochs", metavar="", help="模型保存间隔(单位: 迭代轮数)。默认为1")
        arg_parse.add_argument("--save_dir", dest="save_dir", default="{}".format(self.__save_dir_path),
                               metavar="", help="模型保存路径。默认为 {}".format(self.__save_dir_path))
        arg_parse.add_argument("--dataset", dest="dataset", default="",
                               metavar="", help="数据集目录，默认 {}".format(self.__dataset_path))
        arg_parse.add_argument("--model", dest="model", default="",
                               metavar="", help="PaddleX 模型名称")
        arg_parse.add_argument("--pretrain_weights", dest="pretrain_weights", default="IMAGENET",
                               metavar="", help="从文件加载模型权重，默认 IMAGENET 自动下载 ImageNet 预训练的模型权重")
        arg_parse.add_argument("--resume_checkpoint", dest="resume_checkpoint", default="",
                               metavar="", help="恢复训练时指定上次训练保存的模型路径, 默认不会恢复训练")
        arg_parse.add_argument("--model_list", action="store_true", dest="model_list",
                               help="输出 PaddleX 模型名称，默认不输出，选择后只输出信息，不会开启训练")
        arg_parse.add_argument("--train_list", dest="train_list", default="{}".format(self.__train_list_path),
                               metavar="", help="训练集列表，默认 {}".format(self.__train_list_path))
        arg_parse.add_argument("--eval_list", dest="eval_list", default="{}".format(self.__eval_list_path),
                               metavar="", help="评估集列表，默认 {}".format(self.__eval_list_path))
        arg_parse.add_argument("--label_list", dest="label_list", default="{}".format(self.__label_list_path),
                               metavar="", help="分类标签列表，默认{}".format(self.__label_list_path))
        return arg_parse.parse_args()

    def check(self):
        mod.utils.check_path(self.dataset)
        mod.utils.check_path(self.train_list)
        mod.utils.check_path(self.eval_list)
        mod.utils.check_path(self.label_list)

        # 模型权重
        self.pretrain_weights = None
        # 加载模型权重
        if (self.args.pretrain_weights == ""):
            self.pretrain_weights = None
        elif self.args.pretrain_weights == "IMAGENET":
            self.pretrain_weights = "IMAGENET"
        else:
            mod.utils.check_path(self.args.pretrain_weights)
            self.pretrain_weights = self.args.pretrain_weights

        # 恢复训练时指定上次训练保存的模型路径
        self.resume_checkpoint = None
        # 恢复训练
        if (self.args.resume_checkpoint != ""):
            mod.utils.check_path(self.args.resume_checkpoint)
            self.pretrain_weights = None
            self.resume_checkpoint = self.args.resume_checkpoint


class TestX():
    """
    返回 PaddleX 测试命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 test_list_path=config.TEST_LIST_PATH):
        self.__dataset_path = dataset_path
        self.__test_list_path = test_list_path

        self.args = self.parse() if args == None else args
        self.cpu = self.args.cpu
        self.epochs = self.args.epochs

        self.dataset = dataset_path if self.args.dataset == "" else os.path.join(
            dataset_path, self.args.dataset)
        self.test_list = os.path.join(self.dataset, self.args.test_list)
        self.model_dir = self.args.model_dir

    def parse(self):
        """
        返回命令行参数

        Returns:
            argparse: 命令行参数
        """
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("--cpu", action="store_true",
                               dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        arg_parse.add_argument("--epochs", type=int, default=4,
                               dest="epochs", metavar="", help="训练几轮，默认 4 轮")
        arg_parse.add_argument("--dataset", dest="dataset", default="",
                               metavar="", help="数据集目录，默认 {}".format(self.__dataset_path))
        arg_parse.add_argument("--test_list", dest="test_list", default="{}".format(self.__test_list_path),
                               metavar="", help="训练集列表，默认 {}".format(self.__test_list_path))
        arg_parse.add_argument("--model_dir", dest="model_dir", default="./output/best_model",
                               metavar="", help="读取训练后的模型目录，默认 ./output/best_model")
        return arg_parse.parse_args()

    def check(self):
        mod.utils.check_path(self.dataset)
        mod.utils.check_path(self.test_list)
        mod.utils.check_path(self.model_dir)


class PredictX():
    """
    返回 PaddleX 预测命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 infer_list_path=config.INFER_LIST_PATH):
        self.__dataset_path = dataset_path
        self.__infer_list_path = infer_list_path

        self.args = self.parse() if args == None else args
        self.cpu = self.args.cpu

        self.dataset = dataset_path if self.args.dataset == "" else os.path.join(
            dataset_path, self.args.dataset)
        self.infer_list = os.path.join(self.dataset, self.args.infer_list)
        self.model_dir = self.args.model_dir
        self.result_info = self.args.result_info
        self.result_path = self.args.result_path
        self.split = self.args.split

    def parse(self):
        """
        返回命令行参数

        Returns:
            argparse: 命令行参数
        """
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("--cpu", action="store_true",
                               dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        arg_parse.add_argument("--dataset", dest="dataset", default="",
                               metavar="", help="数据集目录，默认 {}".format(self.__dataset_path))
        arg_parse.add_argument("--infer_list", dest="infer_list", default="{}".format(self.__infer_list_path),
                               metavar="", help="预测集列表，默认 {}".format(self.__infer_list_path))
        arg_parse.add_argument("--model_dir", dest="model_dir", default="./output/best_model",
                               metavar="", help="读取训练后的模型目录，默认 ./output/best_model")
        arg_parse.add_argument("--result_info", action="store_true",
                               dest="result_info", help="显示预测结果详细信息，默认 不显示")
        arg_parse.add_argument("--result_path", dest="result_path", default="./result/result.csv",
                               metavar="", help="预测结果文件路径，默认 ./result/result.csv")
        arg_parse.add_argument("--split", dest="split", default=",",
                               metavar="", help="结果分隔符，默认 ','")
        return arg_parse.parse_args()

    def check(self):
        mod.utils.check_path(self.dataset)
        mod.utils.check_path(self.infer_list)
        mod.utils.check_path(self.model_dir)
