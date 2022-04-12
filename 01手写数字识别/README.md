# 小熊飞桨练习册-01手写数字识别

## 简介

小熊百度飞桨练习项目，01手写数字识别，本项目开发和测试均在 Ubuntu20.04 系统下进行。  
项目最新代码查看主页：[小熊飞桨练习册](https://gitee.com/cnhemiya/paddle-workbook)。  
百度飞桨 AI Studio 主页：[小熊飞桨练习册-01手写数字识别](https://aistudio.baidu.com/aistudio/projectdetail/3796241)

## 文件说明

|文件|说明|
|--|--|
|train.py|训练程序|
|test.py|测试程序|
|report.py|报表程序|
|aistudio-data.sh|aistudio 环境数据整理程序|
|mod/lenet.py|LeNet 网络模型|
|mod/dataset.py|MNIST 手写数据集解析|
|mod/utils.py|杂项|
|mod/config.py|配置|
|mod/report.py|结果报表|
|dataset|数据集目录|
|params|模型参数保存目录|

## 数据集

数据集来源于百度飞桨公共数据集：[经典MNIST数据集](https://aistudio.baidu.com/aistudio/datasetdetail/65)。  
如果运行在本地计算机下载完数据集后解压文件放到 **dataset** 目录下即可。  
如果运行在百度 **aistudio** 环境查看 **data** 目录有 **.gz** 数据在项目根目录运行下面命令即可。
```bash
bash aistudio-data.sh
```

## 网络模型

网络模型使用 **LeNet 网络模型** 来源百度飞桨教程和网络

```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# LeNet 网络模型
class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        if num_classes < 1:
            raise Exception("分类数量 num_classes 必须大于 0: {}".format(num_classes))
        self.num_classes = num_classes
        self.conv1 = nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(
            in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2D(
            in_channels=16, out_channels=120, kernel_size=4, stride=1)
        self.fc1 = nn.Linear(in_features=120, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

## 数据集解析

数据集解析方法来源百度飞桨教程和网络，和百度飞桨 MNIST 数据集稍有不同

```python
import paddle
import os
import struct
import numpy as np


class MNIST(paddle.io.Dataset):
    """
    步骤一: 继承paddle.io.Dataset类
    """

    def __init__(self,
                 image_path=None,
                 label_path=None,
                 transform=None,
                 ):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MNIST, self).__init__()
        self.image_path = image_path
        self.label_path = label_path
        self._check_path(image_path, "数据路径错误")
        self._check_path(label_path, "标签路径错误")
        self.transform = transform
        self.images, self.labels = self.parse_dataset(image_path, label_path)

    def __getitem__(self, idx):
        """
        获取单个数据和标签
        """
        image, label = self.images[idx], self.labels[idx]
        # 这里 reshape 是2维 [28 ,28]
        image = np.reshape(image, [28, 28])
        if self.transform is not None:
            image = self.transform(image)
        # label.astype 如果是整型，只能是 int64
        return image.astype('float32'), label.astype('int64')

    def __len__(self):
        return len(self.labels)

    def _check_path(self, path, msg):
        if not os.path.exists(path):
            raise Exception("{}: {}".format(msg, path))

    @staticmethod
    def parse_dataset(image_path, label_path):
        """
        数据集解析
        """
        with open(image_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            # 这里 reshape 是1维 [786]
            images = np.fromfile(
                imgpath, dtype=np.uint8).reshape(num, rows * cols)
        with open(label_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        return images, labels
```

## 配置模块

可以自行修改 **mod/config.py** 文件，有详细的说明

## 开始训练

运行 **train.py** 文件，查看命令行参数加 -h

```bash
python3 train.py
```

```bash
  --cpu             是否使用 cpu 计算，默认使用 CUDA
  --learning-rate   学习率，默认 0.001
  --epochs          训练几轮，默认 2 轮
  --batch-size      一批次数量，默认 128
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
  --batch-size    一批次数量，默认 128
  --num-workers   线程数量，默认 2
  --load-dir      读取模型参数，读取 params 目录下的子文件夹, 默认 best 目录
```

## 查看结果报表

运行 **report.py** 文件，可以显示 **params** 目录下所有子目录的 **report.json**，  
然后根据 **loss** 最小的模型参数保存在 **best** 子目录下。

```bash
python3 report.py
```

## report.json 说明

|键|说明|
|--|--|
|id|根据模型保存时间生成的 id|
|loss|本次训练的 loss 值|
|acc|本次训练的 acc 值|
|epochs|程序命令行传入的 epochs 值|
|batch_size|程序命令行传入的 batch_size 值|
|learning_rate|程序命令行传入的 learning_rate 值|
