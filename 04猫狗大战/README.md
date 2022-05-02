# 小熊飞桨练习册-04猫狗大战

## 简介

小熊飞桨练习册-04猫狗大战，本项目开发和测试均在 Ubuntu 20.04 系统下进行。  
项目最新代码查看主页：[小熊飞桨练习册](https://gitee.com/cnhemiya/paddle-workbook)  
百度飞桨 AI Studio 主页：[小熊飞桨练习册-04猫狗大战](https://aistudio.baidu.com/aistudio/projectdetail/3925261)  
Ubuntu 系统安装 CUDA 参考：[Ubuntu 百度飞桨和 CUDA 的安装](https://my.oschina.net/hemiya/blog/5509991)

## 文件说明

|文件|说明|
|--|--|
|train.py|训练程序|
|test.py|测试程序|
|test-gtk.py|测试程序 GTK 界面|
|report.py|报表程序|
|onekey.sh|一键获取数据到 dataset 目录下|
|get-data.sh|获取数据到 dataset 目录下|
|make-images-labels.py|生成训练集图像路径和标签的文本文件|
|make-test.py|生成测试集图像路径和标签的文本文件|
|check-data.sh|检查 dataset 目录下的数据是否存在|
|mod/googlenet.py|GoogLeNet 网络模型|
|mod/dataset.py|ImageClass 图像分类数据集解析|
|mod/utils.py|杂项|
|mod/config.py|配置|
|mod/report.py|结果报表|
|dataset|数据集目录|
|params|模型参数保存目录|
|log|VisualDL 日志保存目录|

## 数据集

数据集来源于百度飞桨公共数据集：[猫狗大战-学习](https://aistudio.baidu.com/aistudio/datasetdetail/20743)

### 一键获取数据

- 运行脚本，包含以下步骤：获取数据，生成图像路径和标签的文本文件，检查数据。

如果运行在本地计算机，下载完数据，文件放到 **dataset** 目录下，在项目目录下运行下面脚本。  
如果运行在百度 **AI Studio** 环境，查看 **data** 目录是否有数据，在项目目录下运行下面脚本。

```bash
bash onekey.sh
```

#### 获取数据

如果运行在本地计算机，下载完数据，文件放到 **dataset** 目录下，在项目目录下运行下面脚本。  
如果运行在百度 **AI Studio** 环境，查看 **data** 目录是否有数据，在项目目录下运行下面脚本。

```bash
bash get-data.sh
```

#### 生成图像路径和标签的文本文件

获取数据后，在项目目录下运行下面脚本，生成图像路径和标签的文本文件，包含：

- 训练集 train-images-labels.txt
- 测试集 test-images-labels.txt

```bash
python3 make-images-labels.py train ./dataset catVSdog/train/cat 0 catVSdog/train/dog 1
python3 make-test.py
```

#### 分类标签

- 猫 0
- 狗 1

#### 检查数据

获取数据完毕后，在项目目录下运行下面脚本，检查 dataset 目录下的数据是否存在。

```bash
bash check-data.sh
```

## 网络模型

网络模型使用 **GoogLeNet 网络模型** 来源百度飞桨教程和网络。  
**GoogLeNet 网络模型** 参考： [百度飞桨教程](https://www.paddlepaddle.org.cn/tutorials/projectdetail/3106582)

```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Inception(nn.Layer):
    """
    Inception模块
    """

    def __init__(self, c0, c1, c2, c3, c4):
        """
        Inception模块

        Args:
            c0 (int): 模块输入通道数
            c1 (int): 第一支路 1x1 卷积输出通道数
            c2 (list | tuple): 第二支路 1x1 卷积输入通道数 , 3x3 卷积输出通道数
            c3 (list | tuple): 第三支路 1x1 卷积输入通道数 , 5x5 卷积输出通道数
            c4 (int): 第四支路 3x3 池化 , 1x1 卷积输出通道数
        """
        super(Inception, self).__init__()

        # 第一支路 1x1 卷积输出通道数
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=c0, out_channels=c1,
                      kernel_size=1, stride=1),
            nn.ReLU())

        # 第二支路 1x1 卷积输入通道数 , 3x3 卷积输出通道数
        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=c0,
                      out_channels=c2[0], kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=c2[0], out_channels=c2[1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        # 第三支路 1x1 卷积输入通道数 , 5x5 卷积输出通道数
        self.block3 = nn.Sequential(
            nn.Conv2D(in_channels=c0,
                      out_channels=c3[0], kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=c3[0], out_channels=c3[1],
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU())

        # 第四支路 3x3 池化 , 1x1 卷积输出通道数
        self.block4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=c0, out_channels=c4,
                      kernel_size=1, stride=1),
            nn.ReLU())

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(x)
        b3 = self.block3(x)
        b4 = self.block4(x)
        out = [b1, b2, b3, b4]
        return paddle.concat(out, axis=1)


# GoogLeNet 网络模型
class GoogLeNet(nn.Layer):
    """
    GoogLeNet 网络模型

    输入图像大小为 224 x 224
    """

    def __init__(self, num_classes=10):
        """
        GoogLeNet 网络模型

        Args:
            num_classes (int, optional): 分类数量, 默认 10

        Raises:
            Exception: 分类数量 num_classes 必须大于等于 2
        """
        super(GoogLeNet, self).__init__()
        if num_classes < 2:
            raise Exception(
                "分类数量 num_classes 必须大于等于 2: {}".format(num_classes))
        self.num_classes = num_classes

        # 数据输入处理块
        self.block0 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64,
                      kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=192,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1))

        # 第三个模块包含2个Inception块
        self.block3_1 = Inception(192, 64, [96, 128], [16, 32], 32)
        self.block3_2 = Inception(256, 128, [128, 192], [32, 96], 64)
        self.pool3 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第四个模块包含5个Inception块
        self.block4_1 = Inception(480, 192, [96, 208], [16, 48], 64)
        self.block4_2 = Inception(512, 160, [112, 224], [24, 64], 64)
        self.block4_3 = Inception(512, 128, [128, 256], [24, 64], 64)
        self.block4_4 = Inception(512, 112, [144, 288], [32, 64], 64)
        self.block4_5 = Inception(528, 256, [160, 320], [32, 128], 128)
        self.pool4 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第五个模块包含2个Inception块
        self.block5_1 = Inception(832, 256, [160, 320], [32, 128], 128)
        self.block5_2 = Inception(832, 384, [192, 384], [48, 128], 128)
        self.avg_pool5 = nn.AvgPool2D(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc5 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        # 数据输入处理块
        x = self.block0(x)

        # # 第三个模块包含2个Inception块
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.pool3(x)

        # # # 第四个模块包含5个Inception块
        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)
        x = self.block4_4(x)
        x = self.block4_5(x)
        x = self.pool4(x)

        # # 第五个模块包含2个Inception块
        x = self.block5_1(x)
        x = self.block5_2(x)
        x = self.avg_pool5(x)
        # flatten 根据给定的 start_axis 和 stop_axis 将连续的维度展平
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.dropout(x)
        x = self.fc5(x)
        x = F.softmax(x)

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
        if not os.path.exists(image_path):
            raise Exception("{}: {}".format("图像路径错误", image_path))
        ppvs.set_image_backend("pil")
        # 统一转为 3 通道, png 是 4通道
        image = Image.open(image_path).convert("RGB")
        if transform is not None:
            image = transform(image)
        # 转换图像 HWC 转为 CHW
        # image = np.transpose(image, (2, 0, 1))
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
            if (len(data) < 2):
                raise Exception("数据集解析错误，数据少于 2")
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
python3 test-gtk.py
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

运行 **report.py** 文件，可以显示 **params** 目录下所有子目录的 **report.json**。  
加参数 **--best** 根据 **loss** 最小的模型参数保存在 **best** 子目录下。

```bash
python3 report.py
```

## report.json 说明

|键名|说明|
|--|--|
|id|根据时间生成的字符串 ID|
|loss|本次训练的 loss 值|
|acc|本次训练的 acc 值|
|epochs|本次训练的 epochs 值|
|batch_size|本次训练的 batch_size 值|
|learning_rate|本次训练的 learning_rate 值|

## VisualDL 可视化分析工具

- 安装和使用说明参考：[VisualDL](https://gitee.com/paddlepaddle/VisualDL)
- 训练的时候加上参数 **--log**
- 如果是 **AI Studio** 环境训练的把 **log** 目录下载下来，解压缩后放到本地项目目录下 **log** 目录
- 在项目目录下运行下面命令
- 然后根据提示的网址，打开浏览器访问提示的网址即可

```bash
visualdl --logdir ./log
```
