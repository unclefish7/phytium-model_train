import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_model(num_classes):
    """创建 MobileNetV3 模型"""
    model = models.mobilenet_v3_small(pretrained=True)
    # 替换最后一层
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def get_test_loader(test_dir, batch_size=32):
    """创建测试数据加载器"""
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader, test_dataset.classes


def load_model(model_path, device):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    best_val_acc = checkpoint.get('best_val_acc', 0)
    
    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功，类别数: {num_classes}, 最佳验证准确率: {best_val_acc:.2f}%")
    return model, num_classes


def predict(model, test_loader, device):
    """进行预测"""
    all_predictions = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)


def calculate_metrics(y_true, y_pred, num_classes):
    """计算各种评估指标"""
    # 计算准确率
    accuracy = np.mean(y_true == y_pred)
    
    # 计算每个类别的TP, FP, FN
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp[i] = np.sum((y_true == i) & (y_pred == i))
        fp[i] = np.sum((y_true != i) & (y_pred == i))
        fn[i] = np.sum((y_true == i) & (y_pred != i))
    
    # 计算精确率、召回率和F1分数
    precision_per_class = tp / (tp + fp + 1e-8)
    recall_per_class = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-8)
    
    # 计算加权平均（按类别样本数量加权）
    class_counts = np.bincount(y_true, minlength=num_classes)
    total_samples = len(y_true)
    
    precision = np.sum(precision_per_class * class_counts) / total_samples
    recall = np.sum(recall_per_class * class_counts) / total_samples
    f1_score = np.sum(f1_per_class * class_counts) / total_samples
    
    # 计算混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    
    # 创建热力图
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(im)
    
    # 设置标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在每个格子中添加数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵已保存至: {save_path}")


def print_detailed_results(metrics, class_names):
    """打印详细的评估结果"""
    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)
    
    print(f"整体准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"加权精确率 (Precision): {metrics['precision']:.4f}")
    print(f"加权召回率 (Recall): {metrics['recall']:.4f}")
    print(f"加权F1分数 (F1-Score): {metrics['f1_score']:.4f}")
    
    print("\n各类别详细指标:")
    print("-" * 70)
    print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        precision = metrics['precision_per_class'][i]
        recall = metrics['recall_per_class'][i]
        f1 = metrics['f1_per_class'][i]
        print(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description='测试图像分类器')
    parser.add_argument('--test-dir', type=str, required=True, help='测试数据路径')
    parser.add_argument('--model-path', type=str, default='best.pth', help='模型文件路径')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--save-cm', type=str, default='confusion_matrix.png', help='混淆矩阵保存路径')
    
    args = parser.parse_args()
    
    # 检查测试数据路径
    if not os.path.exists(args.test_dir):
        raise ValueError(f"测试数据路径不存在: {args.test_dir}")
    
    # 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, num_classes = load_model(args.model_path, device)
    
    # 创建测试数据加载器
    test_loader, class_names = get_test_loader(args.test_dir, args.batch_size)
    print(f"测试数据类别: {class_names}")
    print(f"测试样本数量: {len(test_loader.dataset)}")
    
    # 进行预测
    predictions, true_labels = predict(model, test_loader, device)
    
    # 计算评估指标
    metrics = calculate_metrics(true_labels, predictions, num_classes)
    
    # 打印详细结果
    print_detailed_results(metrics, class_names)
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, args.save_cm)
    
    print("\n测试完成！")


if __name__ == '__main__':
    main()