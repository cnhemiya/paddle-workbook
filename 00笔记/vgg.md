# VGG 网络模型介绍

## 介绍

VGG是当前最流行的CNN模型之一，2014年由Simonyan和Zisserman提出，其命名来源于论文作者所在的实验室Visual Geometry Group。AlexNet模型通过构造多层网络，取得了较好的效果，但是并没有给出深度神经网络设计的方向。VGG通过使用一系列大小为3x3的小尺寸卷积核和池化层构造深度卷积神经网络，并取得了较好的效果。VGG模型因为结构简单、应用性极强而广受研究者欢迎，尤其是它的网络结构设计方法，为构建深度神经网络提供了方向。

## 特点

VGG 16 是 16 层架构，具有一对卷积层、池化层和最后的全连接层。 VGG 网络是更深的网络和更小的过滤器的想法。 VGGNet 将层数从 AlexNet 的八层增加。现在它有具有 16 到 19 层 VGGNet 变体的模型。一个关键的事情是，这些模型一直保持非常小的过滤器和 3 x 3 卷积层，这基本上是最小的卷积层过滤器大小，它正在查看一点点相邻像素。他们只是保留了这个非常简单的 3 x 3 卷积结构，并在整个网络中进行周期性池化。

VGG 使用较小的过滤器，因为参数较少，并且堆叠更多，而不是使用较大的过滤器。 VGG 有更小、更深的过滤器，而不是大过滤器。它最终具有与只有一个 7 x 7 卷积层相同的有效感受野。

VGGNet 有一个卷积层和一个池化层，还有几个卷积层、池化层、几个卷积层等等。 VGG 架构共有 16 个卷积层和全连接层。在这种情况下，VGG 16 有 16 个，VGG 19 有 19 个，这只是一个非常相似的架构，但其中有更多的卷积层。

## 结构

<img src="image/vgg1.png" width="100%">
<img src="image/vgg2.jpeg" width="50%">

## 示例代码

- 飞桨示例代码

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
