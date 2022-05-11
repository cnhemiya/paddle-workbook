# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-09 17:47
文档说明: 配置
"""


import os
import argparse
import paddle
import paddle.vision.transforms as pptf
import paddlex
import mod.utils
import mod.report


# 分类文本, 按照标签排列
CLASS_TXT = ["石头", "剪子", "布"]

# 分类数量
NUM_CLASSES = 3

# 图像通道 彩色 3, 灰度 1
IMAGE_C = 3
# 图像高
IMAGE_H = 224
# 图像宽
IMAGE_W = 224

# 数据集路径
DATASET_PATH = "./dataset/"
# 训练数据
TRAIN_LIST_PATH = "train_list.txt"
# 评估数据
EVAL_LIST_PATH = "val_list.txt"
# 测试数据
TEST_LIST_PATH = "test_list.txt"
# 标签数据
LABEL_LIST_PATH = "labels.txt"


# 模型参数保存的文件夹
SAVE_DIR = "./output/"
# 最佳参数保存的文件夹
SAVE_BAST_DIR = "best"
# 最佳参数保存的路径
SAVE_BEST_PATH = SAVE_DIR + SAVE_BAST_DIR + "/"
# 模型参数保存的前缀
SAVE_PREFIX = "model"

# 日志保存的路径
LOG_DIR = "./log"

# 报表文件名
REPORT_FILE = "report.json"

# PaddleX 报表文件名
REPORT_X_FILE = "report.json"

# 推理结果路径
INFER_PATH = "./result/"

# 使用的模型名称
MODEL_NAME = "None"


# PaddleX 图像分类模型名称
PDX_CLS_MODEL_NAME = ['PPLCNet', 'PPLCNet_ssld', 'ResNet18', 'ResNet18_vd', 'ResNet34',
                      'ResNet34_vd', 'ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet101',
                      'ResNet101_vd', 'ResNet101_vd_ssld', 'ResNet152', 'ResNet152_vd', 'ResNet200_vd',
                      'DarkNet53', 'MobileNetV1', 'MobileNetV2', 'MobileNetV3_small', 'MobileNetV3_small_ssld',
                      'MobileNetV3_large', 'MobileNetV3_large_ssld', 'Xception41', 'Xception65', 'Xception71',
                      'ShuffleNetV2', 'ShuffleNetV2_swish', 'DenseNet121', 'DenseNet161', 'DenseNet169',
                      'DenseNet201', 'DenseNet264', 'HRNet_W18_C', 'HRNet_W30_C', 'HRNet_W32_C',
                      'HRNet_W40_C', 'HRNet_W44_C', 'HRNet_W48_C', 'HRNet_W64_C', 'AlexNet']

# PaddleX 图像分类模型小写名称
PDX_CLS_MODEL_NAME_LOWER = ['pplcnet', 'pplcnet_ssld', 'resnet18', 'resnet18_vd', 'resnet34',
                            'resnet34_vd', 'resnet50', 'resnet50_vd', 'resnet50_vd_ssld', 'resnet101',
                            'resnet101_vd', 'resnet101_vd_ssld', 'resnet152', 'resnet152_vd', 'resnet200_vd',
                            'darknet53', 'mobilenetv1', 'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_small_ssld',
                            'mobilenetv3_large', 'mobilenetv3_large_ssld', 'xception41', 'xception65', 'xception71',
                            'shufflenetv2', 'shufflenetv2_swish', 'densenet121', 'densenet161', 'densenet169',
                            'densenet201', 'densenet264', 'hrnet_w18_c', 'hrnet_w30_c', 'hrnet_w32_c',
                            'hrnet_w40_c', 'hrnet_w44_c', 'hrnet_w48_c', 'hrnet_w64_c', 'alexnet']


def pdx_cls_model(model_name: str, num_classes: int):
    """
    获取 PaddleX 分类图像模型

    Args:
        model_name (str):  PaddleX 图像分类模型名称
        num_classes (int): 分类数量

    Returns:
        model: 模型
        model_name: 模型名称
    """
    model_list = PDX_CLS_MODEL_NAME_LOWER
    model_lower = model_name.lower()
    model = None
    if model_lower not in model_list:
        raise Exception("PaddleX 模型名称错误")

    model_name = PDX_CLS_MODEL_NAME[PDX_CLS_MODEL_NAME_LOWER.index(
        model_lower)]
    if model_lower == "pplcnet":
        model = paddlex.cls.PPLCNet(num_classes=num_classes)
    elif model_lower == "pplcnet_ssld":
        model = paddlex.cls.PPLCNet_ssld(num_classes=num_classes)
    elif model_lower == "resnet18":
        model = paddlex.cls.ResNet18(num_classes=num_classes)
    elif model_lower == "resnet18_vd":
        model = paddlex.cls.ResNet18_vd(num_classes=num_classes)
    elif model_lower == "resnet34":
        model = paddlex.cls.ResNet34(num_classes=num_classes)
    elif model_lower == "resnet34_vd":
        model = paddlex.cls.ResNet34_vd(num_classes=num_classes)
    elif model_lower == "resnet50":
        model = paddlex.cls.ResNet50(num_classes=num_classes)
    elif model_lower == "resnet50_vd":
        model = paddlex.cls.ResNet50_vd(num_classes=num_classes)
    elif model_lower == "resnet50_vd_ssld":
        model = paddlex.cls.ResNet50_vd_ssld(num_classes=num_classes)
    elif model_lower == "resnet101":
        model = paddlex.cls.ResNet101(num_classes=num_classes)
    elif model_lower == "resnet101_vd":
        model = paddlex.cls.ResNet101_vd(num_classes=num_classes)
    elif model_lower == "resnet101_vd_ssld":
        model = paddlex.cls.ResNet101_vd_ssld(num_classes=num_classes)
    elif model_lower == "resnet152":
        model = paddlex.cls.ResNet152(num_classes=num_classes)
    elif model_lower == "resnet152_vd":
        model = paddlex.cls.ResNet152_vd(num_classes=num_classes)
    elif model_lower == "resnet200_vd":
        model = paddlex.cls.ResNet200_vd(num_classes=num_classes)
    elif model_lower == "darknet53":
        model = paddlex.cls.DarkNet53(num_classes=num_classes)
    elif model_lower == "mobilenetv1":
        model = paddlex.cls.MobileNetV1(num_classes=num_classes, scale=1.0)
    elif model_lower == "mobilenetv2":
        model = paddlex.cls.MobileNetV2(num_classes=num_classes, scale=1.0)
    elif model_lower == "mobilenetv3_small":
        model = paddlex.cls.MobileNetV3_small(
            num_classes=num_classes, scale=1.0)
    elif model_lower == "mobilenetv3_small_ssld":
        model = paddlex.cls.MobileNetV3_small_ssld(
            num_classes=num_classes, scale=1.0)
    elif model_lower == "mobilenetv3_large":
        model = paddlex.cls.MobileNetV3_large(
            num_classes=num_classes, scale=1.0)
    elif model_lower == "mobilenetv3_large_ssld":
        model = paddlex.cls.MobileNetV3_large_ssld(num_classes=num_classes)
    elif model_lower == "xception41":
        model = paddlex.cls.Xception41(num_classes=num_classes)
    elif model_lower == "xception65":
        model = paddlex.cls.Xception65(num_classes=num_classes)
    elif model_lower == "xception71":
        model = paddlex.cls.Xception71(num_classes=num_classes)
    elif model_lower == "shufflenetv2":
        model = paddlex.cls.ShuffleNetV2(num_classes=num_classes, scale=1.0)
    elif model_lower == "shufflenetv2_swish":
        model = paddlex.cls.ShuffleNetV2_swish(
            num_classes=num_classes, scale=1.0)
    elif model_lower == "densenet121":
        model = paddlex.cls.DenseNet121(num_classes=num_classes)
    elif model_lower == "densenet161":
        model = paddlex.cls.DenseNet161(num_classes=num_classes)
    elif model_lower == "densenet169":
        model = paddlex.cls.DenseNet169(num_classes=num_classes)
    elif model_lower == "densenet201":
        model = paddlex.cls.DenseNet201(num_classes=num_classes)
    elif model_lower == "densenet264":
        model = paddlex.cls.DenseNet264(num_classes=num_classes)
    elif model_lower == "hrnet_w18_c":
        model = paddlex.cls.HRNet_W18_C(num_classes=num_classes)
    elif model_lower == "hrnet_w30_c":
        model = paddlex.cls.HRNet_W30_C(num_classes=num_classes)
    elif model_lower == "hrnet_w32_c":
        model = paddlex.cls.HRNet_W32_C(num_classes=num_classes)
    elif model_lower == "hrnet_w40_c":
        model = paddlex.cls.HRNet_W40_C(num_classes=num_classes)
    elif model_lower == "hrnet_w44_c":
        model = paddlex.cls.HRNet_W44_C(num_classes=num_classes)
    elif model_lower == "hrnet_w48_c":
        model = paddlex.cls.HRNet_W48_C(num_classes=num_classes)
    elif model_lower == "hrnet_w64_c":
        model = paddlex.cls.HRNet_W64_C(num_classes=num_classes)
    elif model_lower == "alexnet":
        model = paddlex.cls.AlexNet(num_classes=num_classes)

    return model, model_name


def pdx_cls_model_name():
    """
    PaddleX 图像分类模型名称
    """
    return PDX_CLS_MODEL_NAME


def transform():
    """
    获取 transform 对数据进行转换

    Returns:
        Compose: 转换数据的操作组合
    """
    # Resize: 调整图像大小, Normalize: 图像归一化处理, Transpose: 转换图像 HWC 转为 CHW
    return pptf.Compose([pptf.Resize(size=[IMAGE_H, IMAGE_W]),
                         pptf.Normalize(mean=[127.5, 127.5, 127.5], std=[
                                        127.5, 127.5, 127.5], data_format='HWC'),
                         pptf.Transpose()])


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
    return mod.dataset.ImageClass(dataset_path=DATASET_PATH, images_labels_txt_path=TRAIN_LIST_PATH, transform=transform)


def test_dataset(transform):
    """
    获取测试数据集

    Args:
        transform (Compose): 转换数据的操作组合

    Returns:
        ImageClass: ImageClass 图像分类数据集解析
    """
    return mod.dataset.ImageClass(dataset_path=DATASET_PATH, images_labels_txt_path=TEST_LIST_PATH, transform=transform)


def get_save_dir(save_dir=SAVE_DIR, time_id=mod.utils.time_id()):
    """
    获取 模型输出文件夹

    Args:
        save_dir (str, optional): 输出文件夹, 默认 SAVE_DIR
        time_id (str, optional): 根据时间生成的字符串 ID

    Returns:
        str : 输出文件夹
    """
    return os.path.join(save_dir, time_id)


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


def get_result_file(result_dir=INFER_PATH, time_id=mod.utils.time_id()):
    """
    获取预测结果文件

    Args:
        result_dir (str, optional): 预测结果文件夹, 默认 PREDICT_PATH
        time_id (str, optional): 根据时间生成的字符串 ID

    Returns:
        str : 预测结果文件
    """
    return os.path.join(result_dir, time_id + ".txt")


def save_result(data: list, result: list, result_file: str):
    """
    保存预测结果

    Args:
        data (list): 数据列表
        result (list): paddle.Model.predict 结果类表
        result_file (str): 预测结果保存的文件
    """
    lines = []
    for dat, res in zip(data, result):
        pre_idx = res.index(max(res))
        dat = dat[len(DATASET_PATH):]
        lines.append("{} {}\n".format(dat, pre_idx))
    with open(result_file, "w") as f:
        f.writelines(lines)


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
    print("保存模型参数。。。")
    model.save(os.path.join(save_path, save_prefix))
    print("模型参数保存完毕！")
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
    print("读取模型参数。。。")
    model.load(path=load_path, reset_optimizer=reset_optimizer)
    print("模型参数读取完毕！")


def print_num_classes():
    print("分类数量:  {},  分类文本数量:  {}".format(NUM_CLASSES, len(CLASS_TXT)))


def save_report(save_dir: str, id: str, args=None, eval_result=None):
    """
    保存结果报表

    Args:
        save_path (str): 保存的路径
        id (str): 报表 id
        args (_type_, optional): 命令行参数, 默认 None
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
    report.save(os.path.join(save_dir, REPORT_FILE))


def save_report_x(save_dir: str, id: str, model: str, args=None):
    """
    保存 PaddleX 结果报表

    Args:
        save_path (str): 保存的路径
        id (str): 报表 id
        args (_type_, optional): 命令行参数, 默认 None
    """
    report = mod.report.ReportX()
    report.id = id
    report.model = model
    report.epochs = args.epochs
    report.batch_size = args.batch_size
    report.learning_rate = float(args.learning_rate)
    report.save(os.path.join(save_dir, REPORT_X_FILE))


def train_args():
    """
    返回训练命令行参数

    Returns:
        argparse: 命令行参数
    """
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--cpu", action="store_true",
                           dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
    arg_parse.add_argument("--epochs", type=int, default=2,
                           dest="epochs", metavar="", help="训练几轮，默认 2 轮")
    arg_parse.add_argument("--batch_size", type=int, default=2,
                           dest="batch_size", metavar="", help="一批次数量，默认 2")
    arg_parse.add_argument("--learning_rate", type=float, default=0.025,
                           dest="learning_rate", metavar="", help="学习率，默认 0.025")
    arg_parse.add_argument("--lr_decay_epochs", dest="lr_decay_epochs", default="30 60 90",
                           metavar="", help="默认优化器的学习率衰减轮数。默认为 30 60 90")
    arg_parse.add_argument("--lr_decay_gamma", type=float, default=0.1,
                           dest="lr_decay_gamma", metavar="", help="默认优化器的学习率衰减率。默认为0.1")
    arg_parse.add_argument("--save_interval_epochs", type=int, default=1,
                           dest="save_interval_epochs", metavar="", help="模型保存间隔(单位: 迭代轮数)。默认为1")
    arg_parse.add_argument("--dataset", dest="dataset", default="",
                           metavar="", help="数据集目录，默认 dataset")
    arg_parse.add_argument("--model", dest="model", default="",
                           metavar="", help="PaddleX 模型名称")
    arg_parse.add_argument("--weights", dest="weights", default="IMAGENET",
                           metavar="", help="从文件加载模型权重，默认 IMAGENET 自动下载 ImageNet 预训练的模型权重")
    arg_parse.add_argument("--resume", dest="resume", default="",
                           metavar="", help="恢复训练时指定上次训练保存的模型路径, 默认不会恢复训练")
    arg_parse.add_argument("--model_list", action="store_true", dest="model_list",
                           help="输出 PaddleX 模型名称，默认不输出，选择后只输出信息，不会开启训练")
    arg_parse.add_argument("--time_id", action="store_true", dest="time_id",
                           help="模型参数保存的目录加上时间 ID，默认不加，目录加上时间 ID")
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
    arg_parse.add_argument("--dataset", dest="dataset", default="",
                           metavar="", help="数据集目录，默认 dataset")
    arg_parse.add_argument("--model", dest="model", default="",
                           metavar="", help="从文件加载模型")
    arg_parse.add_argument("--epochs", type=int, default=10,
                           dest="epochs", metavar="", help="测试几轮，默认 10 轮")
    return arg_parse.parse_args()


def predict_args():
    """
    返回预测命令行参数

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
    arg_parse.add_argument("--no-save", action="store_true",
                           dest="no_save", help="是否保存预测结果，默认保存, 选择后不保存预测结果")
    return arg_parse.parse_args()


def user_cude(cuda=True):
    """
    使用 cuda gpu 还是 cpu 运算

    Args:
        cuda (bool, optional): cuda, 默认 True
    """
    paddle.device.set_device(
        "gpu:0") if cuda else paddle.device.set_device("cpu")
