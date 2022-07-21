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

# 检查版本
from parl.utils import logger
from model import Model
import model
from agent import Agent
from replay_memory import ReplayMemory
from parl.algorithms import DQN
import numpy as np
from gym.spaces import Box
import gym
import parl
import paddle
import paddle.vision.transforms as T
# assert paddle.__version__ == "2.3.1", "[Version WARNING] please try `pip install paddlepaddle==2.2.0`"
# assert parl.__version__ == "2.0.4", "[Version WARNING] please try `pip install parl==2.0.3`"
# assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"


LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = int(1e5)  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 64  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 5e-4  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等，原值 0.99
TRAIN_EPISODE = 2000 # 训练多少个episode
SAVE_EPISODE = 20 # 模型保存间隔

OBS_SHAPE = [96, 96] # 观察图像的大小，即图像的高和宽
SAVE_PATH = "./dqn_model.ckpt" # 模型保存路径


class TransformsObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = observation[35:195]  # 裁剪
        transforms = T.Compose([
            T.Resize(size=self.shape),
            T.Grayscale(num_output_channels=1), 
            T.Normalize(mean=[0], std=[255], data_format='HWC'),
            T.Transpose()])
        observation = transforms(observation)
        return observation

# 训练一个episode
def run_train_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    # obs = to_features(obs) # 加入的特征转换
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, info = env.step(action)
        # next_obs = to_features(next_obs) # 加入的特征转换
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        # print("step = {}".format(step))
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        # obs = to_features(obs) # 加入的特征转换
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            steps += 1
            next_obs, reward, done, info = env.step(action)
            # obs = to_features(next_obs) # 加入的特征转换
            obs = next_obs
            total_reward += reward
            if render:
                env.render()
            # if done or steps >= 200:
            if done:
                break
        eval_reward.append(total_reward)
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


def save_model(agent, save_path: str):
    agent.save(save_path)


def main():
    env = build_env()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    logger.info('obs_dim {}x{},  act_dim {}'.format(obs_dim[0], obs_dim[1], act_dim))

    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 根据parl框架构建agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    algorithm = DQN(
        model=model, 
        gamma=GAMMA, 
        lr=LEARNING_RATE)
    agent = Agent(
        algorithm=algorithm,
        act_dim=act_dim,
        e_greed=0.1,  # 有一定概率随机选取动作，探索，原值 0.1
        e_greed_decrement=1e-7)  # 随着训练逐步收敛，探索的程度慢慢降低，原值 0

    # 加载模型并评估
    # if os.path.exists(SAVE_PATH):
    #     agent.restore(SAVE_PATH)
    #     run_evaluate_episodes(agent, env, render=True)
    #     exit()

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    print("start memory warmup size: {}".format(MEMORY_WARMUP_SIZE))
    for i in range(MEMORY_WARMUP_SIZE):
        total_reward = run_train_episode(agent, env, rpm)
        if (i + 1) % SAVE_EPISODE == 0:
           logger.info("episode: {}    e_greed: {}    train reward: {}    rpm: {}".format(
            i + 1, agent.e_greed, total_reward, len(rpm)))
    print("end memory warmup size: {}".format(MEMORY_WARMUP_SIZE))

    # start train
    print("start train episode: {}".format(TRAIN_EPISODE))
    for i in range(TRAIN_EPISODE):
        # train part
        total_reward = run_train_episode(agent, env, rpm)

        # test part    render=True 查看显示效果
        if (i + 1) % SAVE_EPISODE == 0:
            eval_reward = run_evaluate_episodes(agent, env, render=False)
            logger.info("episode: {}    e_greed: {}    test reward: {}    rpm: {}".format(
                i + 1, agent.e_greed, eval_reward, len(rpm)))
            logger.info("episode: {}    save model: {}".format(i + 1, SAVE_PATH))
            save_model(agent, SAVE_PATH)

    # 训练结束，保存模型
    agent.save(SAVE_PATH)
    print("end train episode: {}".format(TRAIN_EPISODE))

def model_summary():
    # 原图 210x160x3，image[35:195]  # 裁剪
    net = model.PaddleModel()
    c = 1
    h = 96
    w = 96
    params_info = paddle.summary(
        net, (1, c, h, w))
    print(params_info)

def build_env():
    env = gym.make('Pong-v0')
    env = TransformsObservation(env, OBS_SHAPE)
    return env   

def test_env():
    env = build_env()
    obs = env.observation_space.shape
    obs_dim = env.reset()
    print(obs_dim.shape)

def test_main():
    env = build_env()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    # logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 根据parl框架构建agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    algorithm = DQN(
        model=model, 
        gamma=GAMMA, 
        lr=LEARNING_RATE)
    agent = Agent(
        algorithm=algorithm,
        act_dim=act_dim,
        e_greed=0.2,  # 有一定概率随机选取动作，探索，原值 0.1
        e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低，原值 0

    run_train_episode(agent, env, rpm)

if __name__ == '__main__':
    main()
    # test_main()
    # test_env()
    # model_summary()