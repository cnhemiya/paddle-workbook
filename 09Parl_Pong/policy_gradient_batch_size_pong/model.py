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

# @parl.remote_class
class Model(parl.Model):
    """ 使用全连接网络.

    参数:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()

        mid_size = 128 # 中间层

        self.model = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=mid_size),
            nn.ReLU(),
            nn.Linear(in_features=mid_size, out_features=mid_size),
            nn.ReLU(),
            nn.Linear(in_features=mid_size, out_features=act_dim),
            nn.Softmax()
        )

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        # obs = paddle.to_tensor(obs, dtype='float32')
        return self.model(obs)
