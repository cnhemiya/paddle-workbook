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
|quant.py|量化程序|
|prune.py|裁剪程序|
|test.py|测试程序|
|infer.py|预测程序|
|onekey.sh|一键获取数据到 dataset 目录下|
|onetasks.sh|一键训练，量化脚本|
|get_data.sh|获取数据到 dataset 目录下|
|check_data.sh|检查 dataset 目录下的数据是否存在|
|mod/args.py|命令行参数解析|
|mod/pdxconfig.py|PaddleX 配置|
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

可以查看修改 **mod/pdxconfig.py** 文件，有详细的说明

## 开始训练

运行 **train.py** 文件，查看命令行参数加 -h

- 示例

```bash
python3 run/train.py \
    --dataset ./dataset/train \
    --epochs 32 \
    --batch_size 16 \
    --learning_rate 0.01 \
    --lr_decay_epochs "16"\
    --lr_decay_gamma 0.25 \
    --model MobileNetV3_large_ssld \
    --pretrain_weights "IMAGENET"
```

- 参数

```bash
  -h, --help            show this help message and exit
  --cpu                 是否使用 cpu 计算，默认使用 CUDA
  --num_workers         线程数量，默认 auto，为CPU核数的一半
  --epochs              训练几轮，默认 4 轮
  --batch_size          一批次数量，默认 16
  --learning_rate       学习率，默认 0.025
  --early_stop          是否使用提前终止训练策略。默认为 False
  --early_stop_patience 
                        当使用提前终止训练策略时，如果验证集精度在early_stop_patience 个 epoch
                        内连续下降或持平，则终止训练。默认为 5
  --save_interval_epochs 
                        模型保存间隔(单位: 迭代轮数)。默认为 1
  --log_interval_steps 
                        训练日志输出间隔（单位：迭代次数）。默认为 10
  --resume_checkpoint   恢复训练时指定上次训练保存的模型路径, 默认不会恢复训练
  --save_dir            模型保存路径。默认为 ./output/
  --dataset             数据集目录，默认 ./dataset/
  --train_list          训练集列表，默认 '--dataset' 参数目录下的 train_list.txt
  --eval_list           评估集列表，默认 '--dataset' 参数目录下的 val_list.txt
  --label_list          分类标签列表，默认 '--dataset' 参数目录下的 labels.txt
  --warmup_steps        默认优化器的 warmup 步数，学习率将在设定的步数内，从 warmup_start_lr
                        线性增长至设定的 learning_rate，默认为 0
  --warmup_start_lr     默认优化器的 warmup 起始学习率，默认为 0.0
  --lr_decay_epochs     默认优化器的学习率衰减轮数。默认为 30 60 90
  --lr_decay_gamma      默认优化器的学习率衰减率。默认为 0.1
  --use_ema             是否使用指数衰减计算参数的滑动平均值。默认为 False
  --opti_scheduler      优化器的调度器，默认 auto，可选 auto，cosine，piecewise
  --opti_reg_coeff      优化器衰减系数，如果 opti_scheduler 是 Cosine，默认是 4e-05，如果
                        opti_scheduler 是 Piecewise，默认是 1e-04
  --pretrain_weights    若指定为'.pdparams'文件时，从文件加载模型权重；若为字符串’IMAGENET’，则自动下载在Ima
                        geNet图片数据上预训练的模型权重；若为字符串’COCO’，则自动下载在COCO数据集上预训练的模型权重；
                        若为None，则不使用预训练模型。默认为'IMAGENET'
  --model               PaddleX 模型名称
  --model_list          输出 PaddleX 模型名称，默认不输出，选择后只输出信息，不会开启训练
  --backbone            目标检测模型的 backbone 网络
```

## 查看支持的模型

- 运行命令

```bash
python3 run/train.py --model_list
```

- 结果

```bash
'PPLCNet', 'PPLCNet_ssld', 'ResNet18', 'ResNet18_vd', 'ResNet34', 'ResNet34_vd', 'ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet101', 'ResNet101_vd', 'ResNet101_vd_ssld', 'ResNet152', 'ResNet152_vd', 'ResNet200_vd', 'DarkNet53', 'MobileNetV1', 'MobileNetV2', 'MobileNetV3_small', 'MobileNetV3_small_ssld', 'MobileNetV3_large', 'MobileNetV3_large_ssld', 'Xception41', 'Xception65', 'Xception71', 'ShuffleNetV2', 'ShuffleNetV2_swish', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 'DenseNet264', 'HRNet_W18_C', 'HRNet_W30_C', 'HRNet_W32_C', 'HRNet_W40_C', 'HRNet_W44_C', 'HRNet_W48_C', 'HRNet_W64_C', 'AlexNet'
```

## 测试模型

运行 **test.py** 文件，查看命令行参数加 -h

- 示例

```bash
python3 run/test.py --model_dir ./output/best_model \
    --epochs 4 \
    --dataset ./dataset/train \
    --test_list ./dataset/train/test_list.txt
```
- 参数

```bash
  -h, --help    show this help message and exit
  --cpu         是否使用 cpu 计算，默认使用 CUDA
  --epochs      测试几轮，默认 4 轮
  --dataset     数据集目录，默认 ./dataset/
  --test_list   训练集列表，默认 '--dataset' 参数目录下的 test_list.txt
  --model_dir   读取训练后的模型目录，默认 ./output/best_model
```

## 预测模型

运行 **infer.py** 文件，查看命令行参数加 -h

- 示例

```bash
python3 run/infer.py --dataset ./dataset/train --model_dir ./output/best_model
```

- 参数

```bash
  -h, --help      show this help message and exit
  --cpu           是否使用 cpu 计算，默认使用 CUDA
  --dataset       数据集目录，默认 ./dataset/
  --infer_list    预测集列表，默认 '--dataset' 参数目录下的 infer_list.txt
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
