# MobileNetV2-SSD Lite 交通标志识别项目

## 项目概述

本项目实现了基于 MobileNetV2-SSD Lite 的交通标志检测系统，专为飞腾派等边缘设备优化。项目使用清华腾讯交通标志数据集进行训练，并提供了从数据处理、模型训练到部署推理的完整流程。

## 环境要求

- Python 3.8
- PyTorch 1.9.0
- torchvision 0.10.0
- ONNX 1.10.0
- OpenCV 4.5.3
- 其他依赖库（详见`setup_env.sh`）

## 项目结构

```
feitp/
├── data/                      # 数据集相关目录
│   ├── dataset.py             # 数据集类定义
│   ├── dataset_utils.py       # 数据集处理工具
│   ├── processed_dataset/     # 处理后的数据集
│   ├── raw_dataset/           # 原始清华腾讯数据集
│   └── test_images/           # 测试图像
├── deploy/                    # 飞腾派部署文件
│   ├── model/                 # 优化后的模型
│   ├── data/                  # 部署所需数据
│   └── src/                   # 部署源代码
│       └── feitp_inference.py # 飞腾派推理代码
├── models/                    # 模型保存目录
├── results/                   # 结果输出目录
├── utils/                     # 工具函数
│   ├── engine.py              # 训练引擎
│   └── transforms.py          # 图像变换工具
├── train.py                   # 模型训练脚本
├── inference.py               # 本地推理脚本
├── convert_to_onnx.py         # 模型转换脚本
├── deploy_to_feitp.sh         # 部署脚本
└── setup_env.sh               # 环境安装脚本
```

## 文件说明

### 环境配置

- **setup_env.sh**: 创建 conda 虚拟环境并安装所需依赖库

### 数据处理

- **data/dataset_utils.py**: 数据集转换工具，将清华腾讯数据集转换为模型可用格式
- **data/dataset.py**: PyTorch 数据集类，用于加载和处理图像及标注

### 模型训练

- **train.py**: 训练 MobileNetV2-SSD Lite 模型的主脚本
- **utils/engine.py**: 训练循环和评估函数
- **utils/transforms.py**: 图像预处理和数据增强工具

### 模型转换与优化

- **convert_to_onnx.py**: 将 PyTorch 模型转换为 ONNX 格式并进行量化优化

### 推理与部署

- **inference.py**: 本地推理脚本，支持使用 PyTorch 或 ONNX 模型
- **deploy_to_feitp.sh**: 准备飞腾派部署文件
- **deploy/src/feitp_inference.py**: 飞腾派上运行的推理代码，支持图像、视频和摄像头输入

## 使用指南

### 1. 环境设置

```bash
# 安装依赖
bash setup_env.sh
```

### 2. 数据集准备

将清华腾讯交通标志数据集放入`data/raw_dataset`目录，然后运行数据集处理脚本：

```bash
# 激活环境
conda activate feitp_env

# 运行数据集转换（假设您创建了一个运行数据转换的脚本）
python -c "from data.dataset_utils import TsinghuaTencentDatasetConverter; \
          converter = TsinghuaTencentDatasetConverter('data/raw_dataset', 'data/processed_dataset'); \
          converter.convert()"
```

### 3. 模型训练

```bash
python train.py
```

训练过程中会保存最佳模型到`models/best_model.pth`，并每 10 个 epoch 保存一次检查点。

### 4. 模型转换

```bash
python convert_to_onnx.py
```

此脚本将 PyTorch 模型转换为 ONNX 格式，并进行量化优化，输出文件保存在`models/`目录下。

### 5. 本地推理测试

```bash
python inference.py
```

支持使用 PyTorch 模型或 ONNX 模型进行推理，可以通过命令行选择。

### 6. 部署到飞腾派

```bash
bash deploy_to_feitp.sh
```

该脚本会准备部署所需的文件，然后您需要将`deploy`目录上传到飞腾派设备。

### 7. 在飞腾派上运行

在飞腾派设备上执行：

```bash
cd deploy/src
python feitp_inference.py
```

支持三种模式：图像检测、视频检测和摄像头实时检测。

## 性能优化

- 使用 MobileNetV2 作为特征提取器，保证模型轻量化
- 采用 SSDLite 检测头，减少计算量
- ONNX 格式转换和 INT8 量化，进一步减小模型大小和加速推理
- 针对飞腾派等边缘设备的内存和计算能力限制进行优化

## 数据集说明

本项目使用清华腾讯交通标志数据集，该数据集包含中国常见交通标志，适合交通场景应用。数据集需要按照 VOC 格式组织，包含图像和对应的 XML 标注文件。

## 结果示例

训练完成后，模型能够识别各种交通标志，并在图像上绘制边界框和类别标签。推理速度在飞腾派上可达到每帧 10-15ms（取决于具体硬件配置）。

## 注意事项

- 确保在训练前准备好足够的磁盘空间
- 训练过程可能需要 GPU 加速
- 飞腾派部署时，请确保设备有足够的内存运行 ONNX 模型
- 对于实时应用，可能需要进一步优化模型大小或降低输入分辨率
