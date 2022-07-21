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

# -*- coding: utf-8 -*-


# 检查版本
from parl.utils import logger
from parl.algorithms import PolicyGradient
from model import Model
from agent import Agent
import numpy as np
import os
import gym
import parl
import paddle
# assert paddle.__version__ == "2.3.1", "[Version WARNING] please try `pip install paddlepaddle==2.2.0`"
# assert parl.__version__ == "2.0.4", "[Version WARNING] please try `pip install parl==2.0.3`"
# assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"


LEARNING_RATE = 5e-4
OBS_DIM_SIZE = [80, 80]
EPISODES = 2000
SAVE_PATH = "./dpg_model.ckpt"
TRAIN_BATCH_SIZE = 32


# 训练一个episode
def run_train_episode(agent, env):
    obs_list, action_list, reward_list = [], [], []
    # for i in range(RUN_TRAIN_RANGE):
    obs = env.reset()
    while True:
        obs = to_features(obs)  # from shape (210, 160, 3) to (100800,)
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = to_features(obs)  # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def to_features(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    # 就是当前场景(obs)的特征提取
    image = image[35:195]  # 裁剪
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 (background type 2)
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    image = np.array(image).astype("float32")
    image = image.ravel()
    return image


def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


def save_model(agent, save_path: str):
    agent.save(save_path)


def main():
    env = gym.make('Pong-v0')
    obs_dim = OBS_DIM_SIZE[0] * OBS_DIM_SIZE[1]  # 80 * 80
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg)

    # 加载模型并评估
    if os.path.exists(SAVE_PATH):
        agent.restore(SAVE_PATH)
    #     run_evaluate_episodes(agent, env, render=True)
    #     exit()

    for i in range(EPISODES):
        # obs_list, action_list, reward_list = run_train_episode(agent, env)

        # if i % 10 == 0:
        #     logger.info("episode {}, reward sum {}.".format(
        #         i, sum(reward_list)))

        # batch_obs = np.array(obs_list)
        # batch_action = np.array(action_list)
        # batch_reward = calc_reward_to_go(reward_list)

        obs_list, action_list, reward_list = run_train_episode(agent, env)

        if (i + 1) % 10 == 0:
            logger.info("episode {}, reward sum {}.".format(
                i + 1, sum(reward_list)))

        batch_obs_list = np.array(obs_list)
        batch_action_list = np.array(action_list)
        batch_reward_list = calc_reward_to_go(reward_list)

        for j in range(TRAIN_BATCH_SIZE - 1):
            obs_list, action_list, reward_list = run_train_episode(agent, env)

            batch_obs = np.array(obs_list)
            batch_action = np.array(action_list)
            batch_reward = calc_reward_to_go(reward_list)

            batch_obs_list = np.append(batch_obs_list, batch_obs, axis=0)
            batch_action_list = np.append(batch_action_list, batch_action, axis=0)
            batch_reward_list = np.append(batch_reward_list, batch_reward, axis=0)
        
        agent.learn(batch_obs_list, batch_action_list, batch_reward_list)

        if (i + 1) % 2 == 0:
            logger.info("episode: {}    save model: {}".format(i + 1, SAVE_PATH))
            save_model(agent, SAVE_PATH)

        if (i + 1) % 2 == 0:
            # render=True 查看显示效果
            total_reward = run_evaluate_episodes(agent, env, render=False)
            logger.info("episode: {}    test reward: {}".format(i + 1, total_reward))
            
    # run_evaluate_episodes(agent, env, render=False)
    
    # 训练结束，保存模型
    save_model(agent, SAVE_PATH)


if __name__ == '__main__':
    # parl.connect("localhost:6006")
    main()
