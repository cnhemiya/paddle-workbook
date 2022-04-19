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

获取数据后，运行下面脚本，生成图像路径和标签的文本文件，包含：

- 训练集 train-images-labels.txt
- 测试集 test-images-labels.txt

```bash
python3 make-images-labels.py
```

### 检查数据

获取数据完毕后，在项目目录下运行下面脚本，检查 dataset 目录下的数据是否存在。

```bash
bash check-data.sh
```

## 网络模型

网络模型使用 **VGG 网络模型** 来源百度飞桨教程和网络。  
**VGG 网络模型** 参考： [百度飞桨教程](https://www.paddlepaddle.org.cn/tutorials/projectdetail/3106582)

```python

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
