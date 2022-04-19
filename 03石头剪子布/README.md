# 小熊飞桨练习册-02眼疾识别

## 简介

小熊百度飞桨练习项目，03石头剪刀布，本项目开发和测试均在 Ubuntu 20.04 系统下进行。  
项目最新代码查看主页：[小熊飞桨练习册](https://gitee.com/cnhemiya/paddle-workbook)  
百度飞桨 AI Studio 主页：[小熊飞桨练习册-03石头剪刀布](https://aistudio.baidu.com/aistudio/projectdetail/3839895)  
Ubuntu 系统安装 CUDA 参考：[Ubuntu 百度飞桨和 CUDA 的安装](https://my.oschina.net/hemiya/blog/5509991)

## 文件说明

|文件|说明|
|--|--|
|train.py|训练程序|
|test.py|测试程序|
|test-gtk.py|测试程序 GTK 界面|
|report.py|报表程序|
|get-data.sh|获取数据到 dataset 目录下|
|make-images-labels.py|生成图像路径和标签的文本文件|
|check-data.sh|检查 dataset 目录下的数据是否存在|
|mod/VGG.py|VGG 网络模型|
|mod/dataset.py|ImageClass 图像分类数据集解析|
|mod/utils.py|杂项|
|mod/config.py|配置|
|mod/report.py|结果报表|
|dataset|数据集目录|
|params|模型参数保存目录|
|log|VisualDL 日志保存目录|

## 数据集

数据集来源于百度飞桨公共数据集：[石头剪刀布](https://aistudio.baidu.com/aistudio/datasetdetail/75404)

### 获取数据

如果运行在本地计算机，下载完数据，文件放到 **dataset** 目录下，在项目目录下运行下面脚本。  
如果运行在百度 **AI Studio** 环境，查看 **data** 目录是否有数据，在项目目录下运行下面脚本。

```bash
bash get-data.sh
```

### 生成图像路径和标签的文本文件

获取数据后，在项目目录下运行下面脚本，生成图像路径和标签的文本文件，包含：

- 训练集 train-images-labels.txt
- 测试集 test-images-labels.txt

```bash
python3 make-images-labels.py ./dataset rps-cv-images/rock 0 rps-cv-images/scissors 1 rps-cv-images/paper 2
```

### 分类标签

- 石头 0
- 剪子 1
- 布 2

### 检查数据

获取数据完毕后，在项目目录下运行下面脚本，检查 dataset 目录下的数据是否存在。

```bash
bash check-data.sh
```

## 网络模型

网络模型使用 **VGG 网络模型** 来源百度飞桨教程和网络。  
**VGG 网络模型** 参考： [百度飞桨教程](https://www.paddlepaddle.org.cn/tutorials/projectdetail/3106582)

```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# VGG 网络模型
class VGG(nn.Layer):
    """
    VGG 网络模型

    输入图像大小为 224 x 224
    """

    def __init__(self, num_classes=10, fc1_in_features=25088):
        """
        VGG 网络模型

        Args:
            num_classes (int, optional): 分类数量, 默认 10
            fc1_in_features (int, optional): 第一层全连接层输入特征数量, 默认 25088, 
                根据 max_pool5 输出结果, 计算得出 512*7*7 = 25088

        Raises:
            Exception: 分类数量 num_classes 必须大于等于 2
        """
        super(VGG, self).__init__()
        if num_classes < 2:
            raise Exception(
                "分类数量 num_classes 必须大于等于 2: {}".format(num_classes))
        self.num_classes = num_classes
        self.fc1_in_features = fc1_in_features

        # 处理块 1
        self.conv1_1 = nn.Conv2D(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2D(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 处理块 2
        self.conv2_1 = nn.Conv2D(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2D(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 处理块 3
        self.conv3_1 = nn.Conv2D(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2D(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2D(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 处理块 4
        self.conv4_1 = nn.Conv2D(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.max_pool4 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 处理块 5
        self.conv5_1 = nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2D(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.max_pool5 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 全连接层 in_features 25088 = max_pool5 输出 512*7*7
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=4096)
        self.drop_ratio1 = 0.5
        self.drop1 = nn.Dropout(self.drop_ratio1)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.drop_ratio2 = 0.5
        self.drop2 = nn.Dropout(self.drop_ratio2)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        # 处理块 1
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.max_pool1(x)

        # 处理块 2
        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.max_pool2(x)

        # 处理块 3
        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = F.relu(x)
        x = self.max_pool3(x)

        # 处理块 4
        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = F.relu(x)
        x = self.max_pool4(x)

        # 处理块 5
        x = self.conv5_1(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = F.relu(x)
        x = self.max_pool5(x)

        # 全连接层
        # flatten 根据给定的 start_axis 和 stop_axis 将连续的维度展平
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc1(x)
        x = F.relu(x)
        # 在全连接之后使用 dropout 抑制过拟合
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x)
        # 在全连接之后使用 dropout 抑制过拟合
        x = self.drop2(x)
        x = self.fc3(x)

        return x

```

## 数据集解析

数据集解析，主要是解析 **图像路径和标签的文本** ，然后根据图像路径读取图像和标签。

```python
import paddle
import os
import random
import numpy as np
from PIL import Image
import paddle.vision as ppvs


class ImageClass(paddle.io.Dataset):
    """
    ImageClass 图像分类数据集解析, 继承 paddle.io.Dataset 类
    """

    def __init__(self,
                 dataset_path: str,
                 images_labels_txt_path: str,
                 transform=None,
                 shuffle=True
                 ):
        """
        构造函数，定义数据集

        Args:
            dataset_path (str): 数据集路径
            images_labels_txt_path (str): 图像和标签的文本路径
            transform (Compose, optional): 转换数据的操作组合, 默认 None
            shuffle (bool, True): 随机打乱数据, 默认 True
        """

        super(ImageClass, self).__init__()
        self.dataset_path = dataset_path
        self.images_labels_txt_path = images_labels_txt_path
        self._check_path(dataset_path, "数据集路径错误")
        self._check_path(images_labels_txt_path, "图像和标签的文本路径错误")
        self.transform = transform
        self.image_paths, self.labels = self.parse_dataset(
            dataset_path, images_labels_txt_path, shuffle)

    def __getitem__(self, idx):
        """
        获取单个数据和标签

        Args:
            idx (Any): 索引

        Returns:
            image (float32): 图像
            label (int): 标签
        """
        image_path, label = self.image_paths[idx], self.labels[idx]
        return self.get_item(image_path, label, self.transform)

    @staticmethod
    def get_item(image_path: str, label: int, transform=None):
        """
        获取单个数据和标签

        Args:
            image_path (str): 图像路径
            label (int): 标签
            transform (Compose, optional): 转换数据的操作组合, 默认 None

        Returns:
            image (float32): 图像
            label (int): 标签
        """
        ppvs.set_image_backend("pil")
        image = Image.open(image_path)
        if transform is not None:
            image = transform(image)
        # 转换图像 HWC 转为 CHW
        image = np.transpose(image, (2, 0, 1))
        return image.astype("float32"), label

    def __len__(self):
        """
        数据数量

        Returns:
            int: 数据数量
        """
        return len(self.labels)

    def _check_path(self, path: str, msg: str):
        """
        检查路径是否存在

        Args:
            path (str): 路径
            msg (str, optional): 异常消息

        Raises:
            Exception: 路径错误, 异常
        """
        if not os.path.exists(path):
            raise Exception("{}: {}".format(msg, path))

    @staticmethod
    def parse_dataset(dataset_path: str, images_labels_txt_path: str, shuffle: bool):
        """
        数据集解析

        Args:
            dataset_path (str): 数据集路径
            images_labels_txt_path (str): 图像和标签的文本路径

        Returns:
            image_paths: 图像路径集
            labels: 分类标签集
        """
        lines = []
        image_paths = []
        labels = []
        with open(images_labels_txt_path, "r") as f:
            lines = f.readlines()
        # 随机打乱数据
        if (shuffle):
            random.shuffle(lines)
        for i in lines:
            data = i.split(" ")
            image_paths.append(os.path.join(dataset_path, data[0]))
            labels.append(int(data[1]))
        return image_paths, labels
```

## 配置模块

可以查看修改 **mod/config.py** 文件，有详细的说明

## 开始训练

运行 **train.py** 文件，查看命令行参数加 -h

```bash
python3 train.py
```

```bash
  --cpu             是否使用 cpu 计算，默认使用 CUDA
  --learning-rate   学习率，默认 0.001
  --epochs          训练几轮，默认 2 轮
  --batch-size      一批次数量，默认 2
  --num-workers     线程数量，默认 2
  --no-save         是否保存模型参数，默认保存, 选择后不保存模型参数
  --load-dir        读取模型参数，读取 params 目录下的子文件夹, 默认不读取
  --log             是否输出 VisualDL 日志，默认不输出
  --summary         输出网络模型信息，默认不输出，选择后只输出信息，不会开启训练
```

## 测试模型

运行 **test.py** 文件，查看命令行参数加 -h

```bash
python3 test.py
```

```bash
  --cpu           是否使用 cpu 计算，默认使用 CUDA
  --batch-size    一批次数量，默认 2
  --num-workers   线程数量，默认 2
  --load-dir      读取模型参数，读取 params 目录下的子文件夹, 默认 best 目录
```

## 测试模型 GTK 界面

运行 **test-gtk.py** 文件，此程序依赖 GTK 库，只能运行在本地计算机。

```bash
python3 test.py
```

### GTK 库安装

```bash
python3 -m pip install pygobject
```

### 使用手册

- 1、点击 **选择模型** 按钮。
- 2、弹出的文件对话框选择模型，模型在 **params** 目录下的子目录的 **model.pdparams** 文件。
- 3、点击 **随机测试** 按钮，就可以看到测试的图像，预测结果和实际结果。 

## 查看结果报表

运行 **report.py** 文件，可以显示 **params** 目录下所有子目录的 **report.json**，  
然后根据 **loss** 最小的模型参数保存在 **best** 子目录下。

```bash
python3 report.py
```

## report.json 说明

|键名|说明|
|--|--|
|id|根据模型保存的时间生成的 id|
|loss|本次训练的 loss 值|
|acc|本次训练的 acc 值|
|epochs|本次训练的 epochs 值|
|batch_size|本次训练的 batch_size 值|
|learning_rate|本次训练的 learning_rate 值|

## VisualDL 可视化分析工具

- 安装和使用说明参考：[VisualDL](https://gitee.com/paddlepaddle/VisualDL)
- 在本地计算机运行，训练的时候加上参数 **--log**
- 在项目目录下运行下面命令
- 然后根据提示的网址，打开浏览器访问提示的网址即可

```bash
visualdl --logdir ./log
```