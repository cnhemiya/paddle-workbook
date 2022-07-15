#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl
import numpy as np


def image_to_tensor(image):
    """
    图像数据转 tensor

    Returns:
        tensor: 转换后的 tensor 数据
    """
    # 图像数据格式 CHW
    shape_size = len(image.shape)
    n, c, h, w = 1, 1, 0, 0
    if shape_size == 1:
        w = image.shape[0]
        data = image.reshape([1, 1, 1, w]).astype("float32")
    elif shape_size == 2:
        h, w = image.shape[0], image.shape[1]
        data = image.reshape([1, 1, h, w]).astype("float32")
    elif shape_size == 3:
        c, h, w = image.shape[0], image.shape[1], image.shape[2]
        data = image.reshape([1, c, h, w]).astype("float32")
    elif shape_size == 4:
        n, c, h, w = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
        data = image.reshape([n, c, h, w]).astype("float32")
    return paddle.to_tensor(data)


class Model(parl.Model):
    """ 使用全连接网络.

    参数:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()

        input_channels = 1
        mid_channels = 32
        first_features = 2240 # 1 c 640, 3 c 3456
        
        self.model = nn.Sequential(
            nn.Conv2D(in_channels=input_channels, out_channels=mid_channels, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=first_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=act_dim)
        )

    def forward(self, obs):
        obs = image_to_tensor(obs)
        obs = self.model(obs)
        return obs


class PaddleModel(nn.Layer):
    """ 使用全连接网络.

    参数:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self):
        super(PaddleModel, self).__init__()
        
        input_channels = 1
        mid_channels = 32
        first_features = 2240 # 1 c 640, 3 c 3456
        
        self.model = nn.Sequential(
            nn.Conv2D(in_channels=input_channels, out_channels=mid_channels, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=first_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=6)
        )

    def forward(self, obs):
        return self.model(obs)