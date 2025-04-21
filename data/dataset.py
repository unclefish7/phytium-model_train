import os
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

class TrafficSignDataset(Dataset):
    def __init__(self, data_dir, image_set, transforms=None):
        """
        交通标志数据集
        Args:
            data_dir: 数据集根目录
            image_set: 'train' 或 'val'
            transforms: 图像变换
        """
        self.data_dir = data_dir
        self.image_set = image_set
        self.transforms = transforms
        
        # 设置图像和标注路径
        self.image_dir = os.path.join(data_dir, "images", image_set)
        self.annotation_dir = os.path.join(data_dir, "annotations", image_set)
        
        # 获取所有图像文件
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 加载类别映射
        with open(os.path.join(data_dir, "classes.txt"), "r") as f:
            self.class_dict = {
                line.strip().split(": ")[1]: int(line.strip().split(": ")[0]) + 1
                for line in f.readlines()
            }
        
        print(f"加载了{len(self.images)}张{image_set}图片，{len(self.class_dict)}个类别")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取图像尺寸
        height, width, _ = img.shape
        
        # 解析XML标注文件
        xml_path = os.path.join(self.annotation_dir, os.path.splitext(self.images[idx])[0] + ".xml")
        target = self._parse_xml(xml_path)
        
        # 转换为tensor
        image_id = torch.tensor([idx])
        boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(target["labels"], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # 构建最终目标字典
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # 应用变换
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def _parse_xml(self, xml_path):
        """解析XML标注文件"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        # 获取图像大小
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        for obj in root.findall("object"):
            # 获取类别
            class_name = obj.find("name").text
            if class_name in self.class_dict:
                label = self.class_dict[class_name]
            else:
                continue  # 跳过未知类别
            
            # 获取边界框
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) / width
            ymin = float(bbox.find("ymin").text) / height
            xmax = float(bbox.find("xmax").text) / width
            ymax = float(bbox.find("ymax").text) / height
            
            # 归一化边界框坐标
            xmin = max(0, min(1, xmin))
            ymin = max(0, min(1, ymin))
            xmax = max(0, min(1, xmax))
            ymax = max(0, min(1, ymax))
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        return {"boxes": boxes, "labels": labels}
