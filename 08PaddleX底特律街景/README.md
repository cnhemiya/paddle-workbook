# 小熊飞桨练习册-08PaddleX底特律街景

## 简介

小熊飞桨练习册-08PaddleX底特律街景，是学习图像分割小项目，本项目开发和测试均在 Ubuntu 20.04 系统下进行。  
项目最新代码查看主页：[小熊飞桨练习册](https://gitee.com/cnhemiya/paddle-workbook)  
百度飞桨 AI Studio 主页：[小熊飞桨练习册-08PaddleX底特律街景](https://aistudio.baidu.com/aistudio/projectdetail/4237728)  
Ubuntu 系统安装 CUDA 参考：[Ubuntu 百度飞桨和 CUDA 的安装](https://my.oschina.net/hemiya/blog/5509991)

- 锯齿狼牙的预测结果，模型：BiSeNetV2

<img src=./doc/visualize_0183.jpg width="80%">
<img src=./doc/visualize_0558.jpg width="80%">

## 文件说明

|文件|说明|
|--|--|
|train.py|训练程序|
|prune.py|裁剪程序|
|quant.py|量化程序|
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
|doc|文档目录|
|output|训练参数保存目录|
|result|预测结果保存目录|

## 环境依赖

- [百度飞桨](https://www.paddlepaddle.org.cn/)
- [PaddleX](https://gitee.com/paddlepaddle/PaddleX)
- **AI Studio** 环境，右侧 **包管理** 手动安装 **PaddleX**

## 数据集

数据集来源于自己收集标注的百度飞桨公共数据集：[锯齿狼牙的底特律街景](https://aistudio.baidu.com/aistudio/datasetdetail/152045)

数据集包含训练集，验证集，测试集，包含 MASK 掩膜 和 COCO 格式数据集，适用图像分割，语义分割，实例分割学习。 

## 如何自己标注数据

- 使用标注工具：[EISeg](https://gitee.com/paddlepaddle/PaddleSeg/tree/release/2.5/EISeg)
- 中文界面，支持 MASK 掩膜 和 COCO 格式

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
    --dataset ./dataset/detroit_streetscape \
    --epochs 32 \
    --batch_size 1 \
    --learning_rate 0.01 \
    --model BiSeNetV2 \
    --pretrain_weights "CITYSCAPES"
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
  --lr_decay_power      默认优化器学习率衰减指数。默认为 0.9
  --use_mixed_loss      是否使用混合损失函数。如果为True，混合使用CrossEntropyLoss和LovaszSoftmaxLoss，权重分别为0
                        .8和0.2。如果为False，则仅使用CrossEntropyLoss。也可以以列表的形式自定义混合损失函数，列表的每一个元素
                        为(损失函数类型，权重)元组，损失函数类型取值范围为['CrossEntropyLoss', 'DiceLoss',
                        'LovaszSoftmaxLoss']。默认为False。
  --align_corners       是网络中对特征图进行插值时是否将四个角落像素的中心对齐。若特征图尺寸为偶数，建议设为True。若特征图尺寸为奇数，建议设为Fal
                        se。默认为False。
  --backbone            图像分割模型 DeepLabV3P 的 backbone 网络，取值范围为['ResNet50_vd',
                        'ResNet101_vd']，默认为'ResNet50_vd'。
  --hrnet_width         图像分割模型 HRNet 的 width 网络，高分辨率分支中特征层的通道数量。默认为48。可选择取值为[18, 48]。
  --pretrain_weights    若指定为'.pdparams'文件时，则从文件加载模型权重；若为字符串'CITYSCAPES'，则自动下载在CITYSCAPES
                        图片数据上预训练的模型权重；若为字符串'PascalVOC'，则自动下载在PascalVOC图片数据上预训练的模型权重；若为字符
                        串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认为'CIT
                        YSCAPES'。
  --model               PaddleX 模型名称
  --model_list          输出 PaddleX 模型名称，默认不输出，选择后只输出信息，不会开启训练
```

## 查看支持的模型

- 运行命令

```bash
python3 run/train.py --model_list
```

- 结果

```bash
PaddleX 图像分割模型
['DeepLabV3P', 'BiSeNetV2', 'UNet', 'HRNet', 'FastSCNN']
DeepLabV3P backbone 网络
['ResNet50_vd', 'ResNet101_vd']
```

## 模型裁剪

- 运行 **prune.py** 文件，查看命令行参数加 -h。
- 注意：有的模型不支持裁剪。
- 裁剪后的精度大部分会降低。
- 参考文档：[模型裁剪](https://gitee.com/paddlepaddle/PaddleX/tree/develop/tutorials/slim/prune)
- 示例

```bash
python3 run/prune.py \
    --dataset ./dataset/detroit_streetscape \
    --epochs 16 \
    --batch_size 1 \
    --learning_rate 0.001 \
    --model_dir ./output/best_model \
    --save_dir ./output/prune \
    --pruned_flops 0.2
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
                        当使用提前终止训练策略时，如果验证集精度在early_stop_patience 个 epoch 内连续下降或持平，则终止训练。默认为 5
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
  --lr_decay_power      默认优化器学习率衰减指数。默认为 0.9
  --model_dir           模型读取路径。默认为 ./output/best_model
  --skip_analyze        是否跳过分析模型各层参数在不同的裁剪比例下的敏感度，默认不跳过
  --pruned_flops        根据选择的 FLOPS 减小比例对模型进行裁剪。默认为 0.2
```

## 模型量化

- 运行 **quant.py** 文件，查看命令行参数加 -h
- model_dir 是正常训练后的模型保存目录。
- 参考文档：[模型量化](https://gitee.com/paddlepaddle/PaddleX/tree/develop/tutorials/slim/quantize)
- 示例

```bash
python3 run/quant.py \
    --dataset ./dataset/detroit_streetscape \
    --epochs 16 \
    --batch_size 1 \
    --learning_rate 0.001 \
    --model_dir ./output/best_model \
    --save_dir ./output/quant
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
                        当使用提前终止训练策略时，如果验证集精度在early_stop_patience 个 epoch 内连续下降或持平，则终止训练。默认为 5
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
  --lr_decay_power      默认优化器学习率衰减指数。默认为 0.9
  --model_dir           模型读取路径。默认为 ./output/best_model
```

## 预测模型

- 运行 **infer.py** 文件，查看命令行参数加 -h
- 示例

```bash
python3 run/infer.py --model_dir ./output/best_model \
    --predict_image ./dataset/detroit_streetscape/0183.jpg
```

- 参数

```bash
  -h, --help            show this help message and exit
  --model_dir           读取模型的目录，默认 './output/best_model'
  --predict_image       预测的图像文件
  --predict_image_dir   预测的图像目录
  --weight              mask可视化结果与原图权重因子，weight表示原图的权重，默认 0.6
  --result_dir          预测结果可视化的保存目录，默认 './result'
```

## 关于图像宽高

- 由于原始图像是 1920x1080 在训练前填充为 1920 的正方形图像，然后调整为 1024x1024，参见 **seg-train.py**

```python
T.Padding(target_size=1920, pad_mode=0, im_padding_value=[0, 0, 0]),
T.Resize(target_size=1024),
```

## 关于预测可视化的结果颜色

- EISeg 颜色通道顺序为 RGB，paddlex.seg.visualize 颜色通道顺序为 BGR

## 部署模型导出

- 图像分割的 --fixed_input_shape 参数无效
- 参考文档：[部署模型导出](https://gitee.com/paddlepaddle/PaddleX/blob/develop/docs/apis/export_model.md)
- 示例

```bash
paddlex --export_inference --model_dir=./output/best_model/ --save_dir=./output/inference_model
```

## VisualDL 可视化分析工具

- 安装和使用说明参考：[VisualDL](https://gitee.com/paddlepaddle/VisualDL)
- 如果是 **AI Studio** 环境训练的把 **output/vdl_log** 目录下载下来，解压缩后放到本地项目目录下 **output/vdl_log** 目录
- 在项目目录下运行下面命令
- 然后根据提示的网址，打开浏览器访问提示的网址即可

```bash
visualdl --logdir ./output/vdl_log
```
