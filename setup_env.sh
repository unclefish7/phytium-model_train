#!/bin/bash

# 创建conda环境
conda create -n feitp_env python=3.8 -y
conda activate feitp_env

# 安装基础依赖
pip install torch==1.9.0 torchvision==0.10.0
pip install opencv-python==4.5.3.56
pip install numpy==1.20.3
pip install matplotlib==3.4.3
pip install tqdm==4.62.3
pip install Pillow==8.3.1
pip install tensorboard==2.6.0
pip install pycocotools==2.0.2
pip install onnx==1.10.0
pip install onnxruntime==1.8.1
pip install thop==0.0.31-2005241907

echo "环境安装完成！"
