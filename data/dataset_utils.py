import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import shutil
import random

class TsinghuaTencentDatasetConverter:
    def __init__(self, dataset_dir, output_dir):
        """
        初始化数据集转换器
        Args:
            dataset_dir: 清华腾讯数据集路径
            output_dir: 输出目录
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.image_dir = os.path.join(dataset_dir, "images")
        self.annotation_dir = os.path.join(dataset_dir, "annotations")
        
        # 创建输出目录
        os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations", "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations", "val"), exist_ok=True)
        
        # 读取所有类别
        self.classes = self._get_classes()
        
    def _get_classes(self):
        """获取数据集中的所有类别"""
        classes = set()
        for xml_file in os.listdir(self.annotation_dir):
            if not xml_file.endswith('.xml'):
                continue
            
            xml_path = os.path.join(self.annotation_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                classes.add(obj.find('name').text)
        
        return sorted(list(classes))
    
    def convert(self, train_ratio=0.8):
        """转换数据集为SSD模型可用格式"""
        # 获取所有图像文件
        all_images = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_images)
        
        # 分割训练集和验证集
        split_idx = int(len(all_images) * train_ratio)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        # 处理训练集
        self._process_subset(train_images, "train")
        
        # 处理验证集
        self._process_subset(val_images, "val")
        
        # 保存类别信息
        with open(os.path.join(self.output_dir, "classes.txt"), "w") as f:
            for i, cls in enumerate(self.classes):
                f.write(f"{i}: {cls}\n")
                
        print(f"转换完成，共{len(self.classes)}个类别，训练集{len(train_images)}张图片，验证集{len(val_images)}张图片")
        
    def _process_subset(self, image_files, subset):
        """处理训练集或验证集"""
        images_output = os.path.join(self.output_dir, "images", subset)
        annotations_output = os.path.join(self.output_dir, "annotations", subset)
        
        for img_file in tqdm(image_files, desc=f"处理{subset}集"):
            img_path = os.path.join(self.image_dir, img_file)
            
            # 复制图像
            shutil.copy(img_path, os.path.join(images_output, img_file))
            
            # 处理XML标注文件
            xml_file = os.path.splitext(img_file)[0] + ".xml"
            xml_path = os.path.join(self.annotation_dir, xml_file)
            
            if os.path.exists(xml_path):
                shutil.copy(xml_path, os.path.join(annotations_output, xml_file))

def create_label_map(classes, output_path):
    """创建标签映射文件"""
    with open(output_path, 'w') as f:
        for i, cls_name in enumerate(classes, 1):  # 从1开始计数，0通常预留给背景
            f.write(f"item {{\n")
            f.write(f"  id: {i}\n")
            f.write(f"  name: '{cls_name}'\n")
            f.write(f"}}\n\n")
    print(f"Label map saved to {output_path}")
