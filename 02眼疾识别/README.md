# 小熊飞桨练习册-02眼疾识别

## 简介

小熊百度飞桨练习项目，02眼疾识别，本项目开发和测试均在 Ubuntu 20.04 系统下进行。  
项目最新代码查看主页：[小熊飞桨练习册](https://gitee.com/cnhemiya/paddle-workbook)  
百度飞桨 AI Studio 主页：[小熊飞桨练习册-02眼疾识别](https://aistudio.baidu.com/aistudio/projectdetail/3830855)  
Ubuntu 系统安装 CUDA 参考：[Ubuntu 百度飞桨和 CUDA 的安装](https://my.oschina.net/hemiya/blog/5509991)

## 文件说明

|文件|说明|
|--|--|
|train.py|训练程序|
|test.py|测试程序|
|test-gtk.py|测试程序 GTK 界面|
|report.py|报表程序|
|get-data.sh|获取数据到 dataset 目录下|
|check-data.sh|检查 dataset 目录下的数据是否存在|
|mod/alexnet.py|AlexNet 网络模型|
|mod/dataset.py|ImageClass 图像分类数据集解析|
|mod/utils.py|杂项|
|mod/config.py|配置|
|mod/report.py|结果报表|
|dataset|数据集目录|
|params|模型参数保存目录|

## 数据集

数据集来源于百度飞桨公共数据集：[眼疾识别数据集iChallenge-整理版](https://aistudio.baidu.com/aistudio/datasetdetail/138865)

### 获取数据

如果运行在本地计算机，下载完数据，文件放到 **dataset** 目录下，在项目目录下运行下面脚本。  
如果运行在百度 **AI Studio** 环境，查看 **data** 目录是否有数据，在项目目录下运行下面脚本。

```bash
bash get-data.sh
```

### 检查数据

获取数据完毕后，在项目目录下运行下面脚本，检查 dataset 目录下的数据是否存在。

```bash
bash check-data.sh
```

## 网络模型

网络模型使用 **AlexNet 网络模型** 来源百度飞桨教程和网络。  
**AlexNet 网络模型** 参考： [百度飞桨教程](https://www.paddlepaddle.org.cn/tutorials/projectdetail/3106582)

```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# AlexNet 网络模型
class AlexNet(nn.Layer):
    """
    AlexNet 网络模型

    输入图像大小为 224 x 224
    池化层 kernel_size = 2, 第一层卷积层填充 paddling = 2
    """
    def __init__(self, num_classes=10, pool_kernel_size=2, conv1_paddling=2, fc1_in_features=9216):
        """
        AlexNet 网络模型

        Args:
            num_classes (int, optional): 分类数量, 默认 10
            pool_kernel_size (int, optional): 池化层核大小, 默认 2
            conv1_paddling (int, optional): 第一层卷积层填充, 默认 2,
                输入图像大小为 224 x 224 填充 2
            fc1_in_features (int, optional): 第一层全连接层输入特征数量, 默认 9216, 
                根据 max_pool3 输出结果, 计算得出 256*6*6 = 9216

        Raises:
            Exception: 分类数量 num_classes 必须大于等于 2
        """        
        super(AlexNet, self).__init__()
        if num_classes < 2:
            raise Exception("分类数量 num_classes 必须大于等于 2: {}".format(num_classes))
        self.num_classes = num_classes
        self.pool_kernel_size = pool_kernel_size
        self.fc1_in_features = fc1_in_features
        self.conv1 = nn.Conv2D(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=conv1_paddling)
        self.max_pool1 = nn.MaxPool2D(kernel_size=pool_kernel_size, stride=2)
        self.conv2 = nn.Conv2D(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.max_pool2 = nn.MaxPool2D(kernel_size=pool_kernel_size, stride=2)
        self.conv3 = nn.Conv2D(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2D(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2D(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2D(kernel_size=pool_kernel_size, stride=2)
        # in_features 9216 = max_pool3 输出 256*6*6
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=4096)
        self.drop_ratio1 = 0.5
        self.drop1 = nn.Dropout(self.drop_ratio1)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.drop_ratio2 = 0.5
        self.drop2 = nn.Dropout(self.drop_ratio2)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.max_pool3(x)
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
                 ):
        """
        构造函数，定义数据集

        Args:
            dataset_path (str): 数据集路径
            images_labels_txt_path (str): 图像和标签的文本路径
            transform (Compose, optional): 转换数据的操作组合, 默认 None
        """
        super(ImageClass, self).__init__()
        self.dataset_path = dataset_path
        self.images_labels_txt_path = images_labels_txt_path
        self._check_path(dataset_path, "数据集路径错误")
        self._check_path(images_labels_txt_path, "图像和标签的文本路径错误")
        self.transform = transform
        self.image_paths, self.labels = self.parse_dataset(dataset_path, images_labels_txt_path)

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
    def get_item(image_path: str, label: int, transform = None):
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
        image = np.transpose(image, (2,0,1))
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
    def parse_dataset(dataset_path: str, images_labels_txt_path: str):
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
