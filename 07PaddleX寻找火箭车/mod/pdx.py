# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-15 20:35
文档说明: PaddleX 用的
"""


import paddlex


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
