#!/bin/bash

# 知识蒸馏训练脚本

# 设置Python路径（如果需要）
# export PYTHONPATH=$PYTHONPATH:/path/to/yolov5

# 基本参数
TEACHER_WEIGHTS="yolov5/runs/train/exp34/weights/best.pt"  # 教师模型（大模型）权重
STUDENT_CONFIG="yolov5/models/yolov5_MobileNetv4_small.yaml"  # 学生模型（小模型）配置
INITIAL_WEIGHTS="best.pt"  # 学生模型初始权重，可以是预训练权重或随机初始化
DATASET="yolov5/data/tt100k.yaml"  # 数据集配置

# 蒸馏参数
TEMPERATURE=4.0  # 蒸馏温度
ALPHA=0.5  # 特征蒸馏权重
BETA=0.5  # 分类蒸馏权重
GAMMA=0.5  # 边界框回归蒸馏权重
DISTILL_WEIGHT=0.5  # 总蒸馏损失权重

# 训练参数
BATCH_SIZE=16
EPOCHS=100
IMG_SIZE=640
DEVICE="0"  # GPU ID，使用CPU则设为"cpu"

# 输出参数
PROJECT="runs/train-distill"
NAME="yolov5n_mobnetv4_small"

# 执行蒸馏训练
python train_distill.py \
    --weights $INITIAL_WEIGHTS \
    --cfg $STUDENT_CONFIG \
    --data $DATASET \
    --teacher-weights $TEACHER_WEIGHTS \
    --temperature $TEMPERATURE \
    --alpha $ALPHA \
    --beta $BETA \
    --gamma $GAMMA \
    --distill-weight $DISTILL_WEIGHT \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --imgsz $IMG_SIZE \
    --device $DEVICE \
    --project $PROJECT \
    --name $NAME
```
