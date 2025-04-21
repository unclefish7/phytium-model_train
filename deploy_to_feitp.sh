#!/bin/bash

# 创建部署目录
mkdir -p deploy/model
mkdir -p deploy/data
mkdir -p deploy/src

# 复制需要的文件
cp models/ssdlite_mobilenet_v2_quantized.onnx deploy/model/
cp data/processed_dataset/classes.txt deploy/data/
cp inference.py deploy/src/

echo "飞腾派部署文件已准备好！"
echo "请将deploy文件夹上传至飞腾派设备。"
