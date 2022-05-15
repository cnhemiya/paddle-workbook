# 小熊飞桨练习册-06Paddlex垃圾分类

## 简介

小熊飞桨练习册-06Paddlex垃圾分类，本项目开发和测试均在 Ubuntu 20.04 系统下进行。  
项目最新代码查看主页：[小熊飞桨练习册](https://gitee.com/cnhemiya/paddle-workbook)  
百度飞桨 AI Studio 主页：[小熊飞桨练习册-06Paddlex垃圾分类](https://aistudio.baidu.com/aistudio/projectdetail/3966896)  
Ubuntu 系统安装 CUDA 参考：[Ubuntu 百度飞桨和 CUDA 的安装](https://my.oschina.net/hemiya/blog/5509991)

## 文件说明

|文件|说明|
|--|--|
|train.py|训练程序|
|test.py|测试程序|
|infer.py|预测程序|
|onekey.sh|一键获取数据到 dataset 目录下|
|get-data.sh|获取数据到 dataset 目录下|
|make-dataset.py|生成数据集列表|
|check-data.sh|检查 dataset 目录下的数据是否存在|
|mod/args.py|命令行参数解析|
|mod/pdx.py|PaddleX 用的|
|mod/config.py|配置|
|mod/utils.py|杂项|
|mod/report.py|结果报表|
|dataset|数据集目录|
|output|训练参数保存目录|
|result|预测结果保存目录|

## 环境依赖

- [百度飞桨](https://www.paddlepaddle.org.cn/)
- [PaddleX](https://gitee.com/paddlepaddle/PaddleX)

## 数据集

数据集来源于百度飞桨公共数据集：[垃圾分类训练集](https://aistudio.baidu.com/aistudio/datasetdetail/33408)

## 一键获取数据

- 运行脚本，包含以下步骤：获取数据，生成图像路径和标签的文本文件，检查数据。
- 详情查看 **onekey.sh**

如果运行在本地计算机，下载完数据，文件放到 **dataset** 目录下，在项目目录下运行下面脚本。

如果运行在百度 **AI Studio** 环境，查看 **data** 目录是否有数据，在项目目录下运行下面脚本。

```bash
bash onekey.sh
```

## 配置模块

可以查看修改 **mod/config.py** 文件，有详细的说明

## 开始训练

运行 **train.py** 文件，查看命令行参数加 -h

- 示例

```bash
python3 train.py --dataset train --epochs 16 \
    --batch_size 64 --learning_rate 0.1 \
    --lr_decay_epochs "4 8 12" --lr_decay_gamma 0.5 \
    --model MobileNetV3_small_ssld
```

- 参数

```bash
  --cpu                 是否使用 cpu 计算，默认使用 CUDA
  --num_workers         线程数量，默认 auto，为CPU核数的一半
  --epochs              训练几轮，默认 4 轮
  --batch_size          一批次数量，默认 16
  --learning_rate       学习率，默认 0.025
  --lr_decay_epochs     默认优化器的学习率衰减轮数。默认为 30 60 90
  --lr_decay_gamma      默认优化器的学习率衰减率。默认为0.1
  --save_interval_epochs 
                        模型保存间隔(单位: 迭代轮数)。默认为1
  --save_dir            模型保存路径。默认为 ./output/
  --dataset             数据集目录，默认 ./dataset/
  --model               PaddleX 模型名称
  --pretrain_weights    从文件加载模型权重，默认 IMAGENET 自动下载 ImageNet 预训练的模型权重
  --resume_checkpoint   恢复训练时指定上次训练保存的模型路径, 默认不会恢复训练
  --model_list          输出 PaddleX 模型名称，默认不输出，选择后只输出信息，不会开启训练
  --train_list          训练集列表，默认 train_list.txt
  --eval_list           评估集列表，默认 val_list.txt
  --label_list          分类标签列表，默认labels.txt
```

## 查看支持的模型

- 运行命令

```bash
python3 train.py --model_list
```

- 结果

```bash
'PPLCNet', 'PPLCNet_ssld', 'ResNet18', 'ResNet18_vd', 'ResNet34', 'ResNet34_vd', 'ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet101', 'ResNet101_vd', 'ResNet101_vd_ssld', 'ResNet152', 'ResNet152_vd', 'ResNet200_vd', 'DarkNet53', 'MobileNetV1', 'MobileNetV2', 'MobileNetV3_small', 'MobileNetV3_small_ssld', 'MobileNetV3_large', 'MobileNetV3_large_ssld', 'Xception41', 'Xception65', 'Xception71', 'ShuffleNetV2', 'ShuffleNetV2_swish', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 'DenseNet264', 'HRNet_W18_C', 'HRNet_W30_C', 'HRNet_W32_C', 'HRNet_W40_C', 'HRNet_W44_C', 'HRNet_W48_C', 'HRNet_W64_C', 'AlexNet'
```

## 测试模型

运行 **test.py** 文件，查看命令行参数加 -h

- 示例

```bash
python3 test.py --dataset train --epochs 4 \
    --model_dir ./output/best_model
```
- 参数

```bash
  --cpu         是否使用 cpu 计算，默认使用 CUDA
  --epochs      训练几轮，默认 4 轮
  --dataset     数据集目录，默认 ./dataset/
  --test_list   训练集列表，默认 test_list.txt
  --model_dir   读取训练后的模型目录，默认 ./output/best_model
```

## 预测模型

运行 **infer.py** 文件，查看命令行参数加 -h

- 示例

```bash
python3 infer.py --dataset train --model_dir ./output/best_model
```

- 参数

```bash
  --cpu           是否使用 cpu 计算，默认使用 CUDA
  --dataset       数据集目录，默认 ./dataset/
  --infer_list    预测集列表，默认 infer_list.txt
  --model_dir     读取训练后的模型目录，默认 ./output/best_model
  --result_info   显示预测结果详细信息，默认 不显示
  --result_path   预测结果文件路径，默认 ./result/result.csv
  --split         数据分隔符，默认 ','
```

## VisualDL 可视化分析工具

- 安装和使用说明参考：[VisualDL](https://gitee.com/paddlepaddle/VisualDL)
- 如果是 **AI Studio** 环境训练的把 **output/vdl_log** 目录下载下来，解压缩后放到本地项目目录下 **output/vdl_log** 目录
- 在项目目录下运行下面命令
- 然后根据提示的网址，打开浏览器访问提示的网址即可

```bash
visualdl --logdir ./output/vdl_log
```
