# yolov5框架的训练命令
```bash
python train.py --batch-size 16 --epochs 100 --name simple_net_finetune_ --cfg models/yolov5_mobilenetv4_small.yaml --data /workspace/dataset/tt100k.yaml --weights /workspace/model/latest_model_20250520/weights/best.pt --device 0 --image-weights --cos-lr --hyp data/hyps/hyp.scratch-low.yaml --img 1024
```

```bash
python train.py --batch-size 4 --epochs 200 --name simple_net_one_cls --cfg models/yolov5_mobilenetv4_small.yaml --data /workspace/dataset/tt100k.yaml --weights '' --device 0 --image-weights --hyp data/hyps/hyp.detect-only.yaml --img 2048
```

```bash
python detect.py --weights runs/train/simple_net_small_dataset_3/weights/best.pt --source /workspace/dataset/images/test  --img 2048
```

```bash
python val.py --weights runs/train/simple_net_small_dataset_3/weights/best.pt --data /workspace/dataset/tt100k.yaml --img 2048 --device 0 --batch-size 4
```

# 自己写的分类器的训练命令

## 训练模型
```bash
# 基本训练命令
python traffic_sign_classifier/train.py --train-dir /workspace/dataset_classify/train --val-dir /workspace/dataset_classify/val --epochs 50 --batch-size 32 --lr 0.001 --model-path best_classifier.pth

# 更长时间训练以获得更好效果
python traffic_sign_classifier/train.py --train-dir /workspace/dataset_classify/train --val-dir /workspace/dataset_classify/val --epochs 100 --batch-size 16 --lr 0.0005 --model-path best_classifier_100epoch.pth

# 小批次训练（适用于显存较小的情况）
python traffic_sign_classifier/train.py --train-dir /workspace/dataset_classify/train --val-dir /workspace/dataset_classify/val --epochs 80 --batch-size 8 --lr 0.001 --model-path best_classifier_small_batch.pth
```

## 测试模型
```bash
# 基本测试命令
python traffic_sign_classifier/test.py --test-dir /workspace/dataset_classify/test --model-path best_classifier.pth --batch-size 32 --save-cm confusion_matrix.png

# 测试特定模型并保存结果到指定位置
python traffic_sign_classifier/test.py --test-dir /workspace/dataset_classify/test --model-path best_classifier_100epoch.pth --batch-size 16 --save-cm results/confusion_matrix_100epoch.png

# 小批次测试
python traffic_sign_classifier/test.py --test-dir /workspace/dataset_classify/test --model-path best_classifier.pth --batch-size 8 --save-cm confusion_matrix_detailed.png
```

## 完整训练+测试流程示例
```bash
# 1. 训练模型
python train.py \
    --train-dir /workspace/dataset_classify \
    --val-dir /workspace/dataset_classify \
    --epochs 100 \
    --batch-size 512 \
    --lr 0.008 \
    --model-path ./models/traffic_sign_best.pth

# 2. 测试模型性能
python test.py \
    --test-dir /workspace/dataset_classify \
    --model-path traffic_sign_best.pth \
    --batch-size 32 \
    --save-cm ./tests/traffic_sign_confusion_matrix.png
```

## 继续训练（Resume Training）
```bash
# 从已有模型继续训练
python train.py \
    --train-dir /workspace/dataset_classify \
    --val-dir /workspace/dataset_classify \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0005 \
    --model-path best_classifier_continued.pth \
    --resume best_classifier.pth

# 从特定epoch的checkpoint继续训练
python traffic_sign_classifier/train.py \
    --train-dir /workspace/dataset_classify/train \
    --val-dir /workspace/dataset_classify/val \
    --epochs 150 \
    --batch-size 16 \
    --lr 0.0001 \
    --model-path best_classifier_final.pth \
    --resume best_classifier_epoch_50.pth
```

## 新增功能说明

### 训练过程中的详细指标
训练过程中每个epoch会输出：
- Train Loss & Accuracy
- Validation Loss & Accuracy  
- Validation Precision & Recall
- Validation mAP@50 & mAP@50:95

### 自动保存功能
- 每10个epoch自动保存checkpoint
- 始终保存最佳验证准确率的模型
- 支持从任何checkpoint继续训练

### 测试详细指标
测试时会输出：
- 整体准确率、精确率、召回率、F1分数
- mAP@50 和 mAP@50:95 指标
- 每个类别的详细指标（精确率、召回率、F1、AP）
- 混淆矩阵可视化