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

## 新的使用方式 - 自动数据分割

### 训练模型（推荐）
```bash
# 基本训练命令 - 自动按8:2分割训练和验证数据
python train.py --data-dir /workspace/dataset_classify --epochs 50 --batch-size 512 --lr 0.008 --model-path ./models/best_classifier.pth

# 自定义分割比例
python traffic_sign_classifier/train.py --data-dir /workspace/dataset_classify --train-ratio 0.7 --val-ratio 0.3 --epochs 100 --batch-size 16 --lr 0.0005 --model-path best_classifier_100epoch.pth

# 小批次训练（适用于显存较小的情况）
python traffic_sign_classifier/train.py --data-dir /workspace/dataset_classify --epochs 80 --batch-size 8 --lr 0.001 --model-path best_classifier_small_batch.pth
```

### 测试模型（推荐）
```bash
# 基本测试命令 - 使用20%的数据作为测试集
python test.py --data-dir /workspace/dataset_classify --model-path ./models/best_classifier.pth --batch-size 32 --save-cm ./test/confusion_matrix.png

# 自定义测试集比例
python traffic_sign_classifier/test.py --data-dir /workspace/dataset_classify --test-ratio 0.15 --model-path best_classifier.pth --save-cm confusion_matrix.png

# 使用剩余数据作为测试集（与训练时使用不同的分割）
python traffic_sign_classifier/test.py --data-dir /workspace/dataset_classify --use-remaining --model-path best_classifier.pth --save-cm confusion_matrix.png
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

## 数据集结构要求

### 新方式（推荐）
只需要一个包含所有数据的目录：
```
dataset_classify/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── class3/
    ├── image1.jpg
    └── ...
```

程序会自动按比例分割数据：
- 默认：80% 训练，20% 验证
- 测试时可以使用剩余数据或指定比例

### 旧方式（仍然支持）
手动分割的目录结构：
```
dataset_classify/
├── train/
│   ├── class1/
│   └── class2/
├── val/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```