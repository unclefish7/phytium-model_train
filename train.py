import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v2
from torchvision.models.detection._utils import BoxCoder
from torchvision.models.detection.ssd import SSD

# 自定义数据集和工具
from data.dataset import TrafficSignDataset
from utils.transforms import get_transform
from utils.engine import train_one_epoch, evaluate

def get_model(num_classes):
    """获取MobileNetV2-SSD Lite模型"""
    # 加载预训练模型
    model = ssdlite320_mobilenet_v2(pretrained=True)
    
    # 修改分类器头以匹配数据集的类别数
    in_channels = model.head.classification_head.in_channels
    num_anchors = model.head.classification_head.num_anchors
    
    # 替换分类头
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        in_channels, num_anchors * num_classes, kernel_size=3, padding=1
    )
    
    return model

def main():
    # 设置参数
    data_dir = os.path.join("data", "processed_dataset")
    output_dir = "models"
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载类别信息
    with open(os.path.join(data_dir, "classes.txt"), "r") as f:
        classes = [line.strip().split(": ")[1] for line in f.readlines()]
    
    num_classes = len(classes) + 1  # +1 用于背景类
    
    # 创建数据集和数据加载器
    train_dataset = TrafficSignDataset(
        data_dir, 
        "train", 
        transforms=get_transform(train=True)
    )
    
    val_dataset = TrafficSignDataset(
        data_dir, 
        "val", 
        transforms=get_transform(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # 获取模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes)
    model.to(device)
    
    # 设置优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, 
        lr=learning_rate, 
        momentum=0.9, 
        weight_decay=0.0005
    )
    
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    
    print(f"开始训练，总共{num_epochs}个epoch")
    
    best_map = 0
    # 训练循环
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            device, 
            epoch, 
            print_freq=10
        )
        
        # 更新学习率
        lr_scheduler.step()
        
        # 评估
        eval_results = evaluate(model, val_loader, device=device)
        mean_ap = eval_results.coco_eval['bbox'].stats[0]
        
        print(f"Epoch {epoch} 验证集 mAP: {mean_ap:.4f}")
        
        # 保存模型
        if mean_ap > best_map:
            best_map = mean_ap
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model.pth"))
            print(f"保存最佳模型，mAP: {best_map:.4f}")
        
        # 每10个epoch保存一次检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_map': best_map
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth"))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    
    print(f"训练完成！最佳mAP: {best_map:.4f}")

if __name__ == "__main__":
    main()
