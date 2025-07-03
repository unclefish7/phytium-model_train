"""
交通标志检测和分类模型评估脚本

依赖安装:
    pip install numpy opencv-python pyyaml matplotlib seaborn scikit-learn onnxruntime torch torchvision

功能描述:
    基于inference.py的推理方法，对模型在YOLO格式数据集上进行全面评估
    - 使用相同的检测+分类两阶段推理流程
    - 支持多种数据增强模拟恶劣天气和环境条件
    - 生成详细的评估指标和可视化结果
    - 输出检测结果示例图片和统计图表

评估流程:
    1. 加载YOLO格式数据集（images + labels）
    2. 对图像应用数据增强（雨雪雾、污渍、划痕等）
    3. 使用inference推理系统进行检测和分类
    4. 计算评估指标（mAP、Precision、Recall等）
    5. 生成可视化结果和示例图片

数据增强类型:
    - 天气模拟：雨滴、雪花、雾气
    - 图像质量：模糊、噪声、亮度变化
    - 物理损坏：污渍、划痕、雨痕
    - 环境条件：光照变化、对比度调整

输出内容:
    - 评估指标报告（控制台输出）
    - 检测结果示例图片（高/中/低置信度）
    - 评估指标可视化图表
    - 混淆矩阵和PR曲线
    - 详细的性能分析报告
"""

import os
import sys
import numpy as np
import cv2
import yaml
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
import onnxruntime as ort
import torch
import torchvision.ops
from sklearn.metrics import confusion_matrix, classification_report

# 添加当前目录到路径以导入inference模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import TrafficSignInference

# ===== 配置常量 =====
# 数据集路径（Docker容器内路径）
DATASET_PATH = "/workspace/dataset_detect_classify"
EVAL_SPLIT = "eval"  # 使用验证集进行评估

# 模型路径
DETECT_MODEL_PATH = "./models/detect_only_LowRes.onnx"
CLASSIFY_MODEL_PATH = "./models/best_classifier.onnx"
LABELS_PATH = "./label.yaml"

# 输出目录
OUTPUT_DIR = "./eval_results"
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
EXAMPLES_DIR = os.path.join(OUTPUT_DIR, "examples")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# 评估参数
IOU_THRESHOLD = 0.5  # IoU阈值用于判断检测是否正确
CONFIDENCE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 用于PR曲线
MAX_EVAL_IMAGES = 1000  # 最大评估图像数量（减少到1000）

# 数据增强参数
AUGMENTATION_PROBABILITY = 0.5  # 降低增强概率
ENABLE_AUGMENTATION = True  # 是否启用数据增强

# 示例图片数量配置
EXAMPLES_PER_CONFIDENCE_LEVEL = 10  # 每个置信度级别的示例数量
GRID_SIZE = (2, 3)  # 示例图片网格大小


class WeatherAugmentation:
    """天气和环境条件数据增强"""
    
    @staticmethod
    def add_rain(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """添加雨滴效果"""
        h, w = image.shape[:2]
        rain_drops = np.random.randint(0, 255, (h, w))
        rain_mask = rain_drops < (intensity * 10)
        
        # 创建雨滴形状
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 8))
        rain_mask = cv2.morphologyEx(rain_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        result = image.copy()
        result[rain_mask > 0] = np.minimum(result[rain_mask > 0] + 50, 255)
        return result
    
    @staticmethod
    def add_snow(image: np.ndarray, intensity: float = 0.2) -> np.ndarray:
        """添加雪花效果"""
        h, w = image.shape[:2]
        snow = np.random.randint(0, 255, (h, w))
        snow_mask = snow < (intensity * 15)
        
        # 创建雪花形状
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        snow_mask = cv2.morphologyEx(snow_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        result = image.copy()
        result[snow_mask > 0] = np.minimum(result[snow_mask > 0] + 80, 255)
        return result
    
    @staticmethod
    def add_fog(image: np.ndarray, intensity: float = 0.4) -> np.ndarray:
        """添加雾气效果"""
        fog_layer = np.full_like(image, 200, dtype=np.uint8)
        alpha = intensity
        result = cv2.addWeighted(image, 1 - alpha, fog_layer, alpha, 0)
        return result
    
    @staticmethod
    def add_blur(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """添加模糊效果"""
        kernel_size = int(5 + intensity * 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def add_dirt_spots(image: np.ndarray, num_spots: int = 5) -> np.ndarray:
        """添加污渍"""
        result = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(num_spots):
            x = random.randint(0, w - 20)
            y = random.randint(0, h - 20)
            size = random.randint(5, 15)
            color = random.randint(50, 150)
            cv2.circle(result, (x, y), size, (color, color, color), -1)
            
        return result
    
    @staticmethod
    def add_scratches(image: np.ndarray, num_scratches: int = 3) -> np.ndarray:
        """添加划痕"""
        result = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(num_scratches):
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            color = random.randint(0, 100)
            thickness = random.randint(1, 3)
            cv2.line(result, (x1, y1), (x2, y2), (color, color, color), thickness)
            
        return result
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float = 0.8) -> np.ndarray:
        """调整亮度"""
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    

    
    @staticmethod
    def add_motion_blur(image: np.ndarray, size: int = 15) -> np.ndarray:
        """添加运动模糊"""
        # 创建运动模糊核
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def simulate_lens_distortion(image: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """模拟镜头畸变（优化版本）"""
        h, w = image.shape[:2]
        
        # 使用numpy向量化操作代替嵌套循环
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        center_x, center_y = w / 2, h / 2
        
        # 向量化计算
        dx = x - center_x
        dy = y - center_y
        r = np.sqrt(dx*dx + dy*dy)
        
        # 应用畸变
        max_dim = max(w, h)
        r_distorted = r * (1 + strength * (r / max_dim)**2)
        
        # 避免除零
        mask = r > 0
        map_x = np.zeros_like(x, dtype=np.float32)
        map_y = np.zeros_like(y, dtype=np.float32)
        
        map_x[mask] = center_x + dx[mask] * r_distorted[mask] / r[mask]
        map_y[mask] = center_y + dy[mask] * r_distorted[mask] / r[mask]
        map_x[~mask] = x[~mask]
        map_y[~mask] = y[~mask]
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    @staticmethod
    def add_shadow(image: np.ndarray, num_shadows: int = 2) -> np.ndarray:
        """添加阴影效果"""
        result = image.copy().astype(np.float32)
        h, w = image.shape[:2]
        
        for _ in range(num_shadows):
            # 随机生成阴影区域
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
            
            # 创建阴影遮罩
            shadow_mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(shadow_mask, (x1, y1), (x2, y2), 1.0, -1)
            
            # 高斯模糊使阴影更自然
            shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 0)
            
            # 应用阴影（降低亮度）
            shadow_factor = random.uniform(0.3, 0.7)
            for c in range(3):
                result[:, :, c] *= (1 - shadow_mask * shadow_factor)
        
        return np.clip(result, 0, 255).astype(np.uint8)


class DatasetLoader:
    """YOLO格式数据集加载器"""
    
    def __init__(self, dataset_path: str, split: str = "val"):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.images_dir = self.dataset_path / "images" / split
        self.labels_dir = self.dataset_path / "labels" / split
        
        # 加载数据集配置
        config_file = self.dataset_path / "data.yaml"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.class_names = self.config['names']
            self.num_classes = self.config['nc']
        else:
            # 使用默认标签文件
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                label_data = yaml.safe_load(f)
            self.class_names = label_data['names']
            self.num_classes = label_data['nc']
    
    def load_annotations(self, image_path: Path) -> List[Dict]:
        """加载图像对应的标注"""
        label_path = self.labels_dir / (image_path.stem + ".txt")
        
        if not label_path.exists():
            return []
        
        annotations = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    annotations.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
        
        return annotations
    
    def get_image_list(self) -> List[Path]:
        """获取图像文件列表"""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in self.images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        # 限制评估图像数量
        if len(image_files) > MAX_EVAL_IMAGES:
            image_files = random.sample(image_files, MAX_EVAL_IMAGES)
        
        return sorted(image_files)


class EvaluationMetrics:
    """评估指标计算器"""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """重置统计数据"""
        self.true_positives = defaultdict(list)
        self.false_positives = defaultdict(list)
        self.false_negatives = defaultdict(list)
        self.all_predictions = []
        self.all_ground_truths = []
        self.confidence_scores = defaultdict(list)
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def convert_yolo_to_xyxy(self, yolo_box: Dict, img_width: int, img_height: int) -> List[float]:
        """将YOLO格式坐标转换为xyxy格式"""
        x_center = yolo_box['x_center'] * img_width
        y_center = yolo_box['y_center'] * img_height
        width = yolo_box['width'] * img_width
        height = yolo_box['height'] * img_height
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return [x1, y1, x2, y2]
    
    def update(self, predictions: List[Dict], ground_truths: List[Dict], 
               img_width: int, img_height: int):
        """更新评估指标"""
        # 转换预测结果格式
        pred_boxes = []
        pred_classes = []
        pred_scores = []
        
        for pred in predictions:
            pred_boxes.append(pred['bbox'])
            pred_classes.append(pred['class_id'])
            pred_scores.append(pred['combined_confidence'])
        
        # 转换真实标注格式
        gt_boxes = []
        gt_classes = []
        
        for gt in ground_truths:
            gt_box = self.convert_yolo_to_xyxy(gt, img_width, img_height)
            gt_boxes.append(gt_box)
            gt_classes.append(gt['class_id'])
        
        # 记录所有预测和真实标注
        self.all_predictions.extend(predictions)
        self.all_ground_truths.extend(ground_truths)
        
        # 匹配预测和真实标注
        matched_gt = set()
        
        for pred_idx, pred in enumerate(predictions):
            pred_box = pred['bbox']
            pred_class = pred['class_id']
            pred_score = pred['combined_confidence']
            
            self.confidence_scores[pred_class].append(pred_score)
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                if gt_classes[gt_idx] != pred_class:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # 判断是否为真正例
            if best_iou >= IOU_THRESHOLD and best_gt_idx != -1:
                self.true_positives[pred_class].append(pred_score)
                matched_gt.add(best_gt_idx)
            else:
                self.false_positives[pred_class].append(pred_score)
        
        # 记录未匹配的真实标注为假负例
        for gt_idx, gt_class in enumerate(gt_classes):
            if gt_idx not in matched_gt:
                self.false_negatives[gt_class].append(1.0)
    
    def calculate_ap(self, class_id: int, confidence_threshold: float = 0.0) -> Tuple[float, float, float]:
        """计算单个类别的AP、Precision和Recall"""
        tp_scores = [s for s in self.true_positives[class_id] if s >= confidence_threshold]
        fp_scores = [s for s in self.false_positives[class_id] if s >= confidence_threshold]
        fn_count = len(self.false_negatives[class_id])
        
        tp_count = len(tp_scores)
        fp_count = len(fp_scores)
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        
        # 计算AP（简化版本，使用固定阈值）
        if tp_count == 0:
            ap = 0.0
        else:
            # 获取所有正例分数并排序
            all_positive_scores = sorted(tp_scores + fp_scores, reverse=True)
            
            precisions = []
            recalls = []
            true_positives_cumsum = 0
            
            for i, score in enumerate(all_positive_scores):
                if score in tp_scores:
                    true_positives_cumsum += 1
                
                p = true_positives_cumsum / (i + 1)
                r = true_positives_cumsum / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
                
                precisions.append(p)
                recalls.append(r)
            
            # 计算AP（梯形积分）
            ap = 0.0
            for i in range(1, len(recalls)):
                ap += (recalls[i] - recalls[i-1]) * precisions[i]
        
        return ap, precision, recall
    
    def calculate_map(self, confidence_threshold: float = 0.0) -> Dict:
        """计算mAP和其他指标"""
        aps = []
        precisions = []
        recalls = []
        class_metrics = {}
        
        for class_id in range(self.num_classes):
            ap, precision, recall = self.calculate_ap(class_id, confidence_threshold)
            aps.append(ap)
            precisions.append(precision)
            recalls.append(recall)
            
            class_metrics[class_id] = {
                'class_name': self.class_names[class_id],
                'ap': ap,
                'precision': precision,
                'recall': recall,
                'tp_count': len([s for s in self.true_positives[class_id] if s >= confidence_threshold]),
                'fp_count': len([s for s in self.false_positives[class_id] if s >= confidence_threshold]),
                'fn_count': len(self.false_negatives[class_id])
            }
        
        mean_ap = np.mean(aps) if aps else 0.0
        mean_precision = np.mean(precisions) if precisions else 0.0
        mean_recall = np.mean(recalls) if recalls else 0.0
        
        f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0.0
        
        return {
            'mAP': mean_ap,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'f1_score': f1_score,
            'class_metrics': class_metrics
        }


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, output_dir: str, class_names: List[str]):
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.examples_dir = self.output_dir / "examples"
        self.metrics_dir = self.output_dir / "metrics"
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
    
    def save_example_images(self, image_results: List[Dict]):
        """保存示例图像"""
        if not image_results:
            return
        
        # 按置信度分组
        high_conf = [r for r in image_results if r['max_confidence'] >= 0.7]
        mid_conf = [r for r in image_results if 0.3 <= r['max_confidence'] < 0.7]
        low_conf = [r for r in image_results if r['max_confidence'] < 0.3]
        
        confidence_groups = {
            'high_confidence': high_conf,
            'medium_confidence': mid_conf,
            'low_confidence': low_conf
        }
        
        for group_name, group_results in confidence_groups.items():
            if not group_results:
                continue
            
            # 随机选择示例
            selected = random.sample(group_results, 
                                   min(EXAMPLES_PER_CONFIDENCE_LEVEL, len(group_results)))
            
            # 创建网格图像
            fig, axes = plt.subplots(GRID_SIZE[0], GRID_SIZE[1], 
                                   figsize=(15, 10))
            fig.suptitle(f'Detection Examples - {group_name.replace("_", " ").title()}\n'
                        f'Green: Predictions, Blue: Ground Truth', 
                        fontsize=16)
            
            axes_flat = axes.flatten() if GRID_SIZE[0] * GRID_SIZE[1] > 1 else [axes]
            
            for idx, result in enumerate(selected[:len(axes_flat)]):
                ax = axes_flat[idx]
                
                # 使用增强后的图像
                if 'augmented_image' in result:
                    image = result['augmented_image'].copy()
                else:
                    # 兼容性：如果没有增强图像，使用原始图像
                    image = cv2.imread(result['image_path'])
                
                if image is not None:
                    # 在图像上绘制检测框
                    for detection in result['detections']:
                        bbox = detection['bbox']
                        class_name = detection['class_name']
                        confidence = detection['combined_confidence']
                        
                        # 绘制检测框
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 添加标签背景
                        label_text = f'{class_name}: {confidence:.2f}'
                        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                        
                        # 添加文本标签
                        cv2.putText(image, label_text, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    # 同时绘制真实标注框（蓝色）
                    if 'ground_truths' in result:
                        img_height, img_width = image.shape[:2]
                        for gt in result['ground_truths']:
                            # 转换YOLO格式到像素坐标
                            x_center = gt['x_center'] * img_width
                            y_center = gt['y_center'] * img_height
                            width = gt['width'] * img_width
                            height = gt['height'] * img_height
                            
                            x1 = int(x_center - width / 2)
                            y1 = int(y_center - height / 2)
                            x2 = int(x_center + width / 2)
                            y2 = int(y_center + height / 2)
                            
                            # 绘制真实标注框（蓝色）
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # 转换BGR到RGB用于matplotlib显示
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ax.imshow(image_rgb)
                
                ax.set_title(f'Max Conf: {result["max_confidence"]:.3f}')
                ax.axis('off')
            
            # 隐藏未使用的子图
            for idx in range(len(selected), len(axes_flat)):
                axes_flat[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.examples_dir / f'{group_name}_examples.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def plot_confidence_distribution(self, image_results: List[Dict]):
        """绘制置信度分布图"""
        confidences = [r['max_confidence'] for r in image_results if r['detections']]
        
        if not confidences:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 直方图
        ax1.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Score Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2.boxplot(confidences, vert=True)
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence Score Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'confidence_distribution.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_class_performance(self, metrics: Dict):
        """绘制各类别性能图"""
        class_metrics = metrics['class_metrics']
        
        if not class_metrics:
            return
        
        class_names = []
        aps = []
        precisions = []
        recalls = []
        
        for class_id, metric in class_metrics.items():
            class_names.append(metric['class_name'])
            aps.append(metric['ap'])
            precisions.append(metric['precision'])
            recalls.append(metric['recall'])
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.bar(x - width, aps, width, label='AP', alpha=0.8)
        ax.bar(x, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x + width, recalls, width, label='Recall', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'class_performance.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_pr_curve(self, metrics: EvaluationMetrics):
        """绘制PR曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 选择几个主要类别绘制PR曲线
        main_classes = list(range(min(4, len(self.class_names))))
        
        for idx, class_id in enumerate(main_classes):
            ax = axes[idx]
            
            # 获取该类别的预测分数
            tp_scores = metrics.true_positives[class_id]
            fp_scores = metrics.false_positives[class_id]
            fn_count = len(metrics.false_negatives[class_id])
            
            if not tp_scores and not fp_scores:
                ax.text(0.5, 0.5, 'No predictions for this class', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{self.class_names[class_id]} - No Data')
                continue
            
            # 计算PR曲线点
            all_scores = sorted(tp_scores + fp_scores, reverse=True)
            precisions = []
            recalls = []
            
            for threshold in all_scores:
                tp = len([s for s in tp_scores if s >= threshold])
                fp = len([s for s in fp_scores if s >= threshold])
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
            
            ax.plot(recalls, precisions, marker='o', markersize=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{self.class_names[class_id]} PR Curve')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'pr_curves.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_metrics_summary(self, metrics: Dict, processing_time: float, 
                           num_images: int, num_detections: int):
        """保存指标摘要"""
        summary = {
            'evaluation_summary': {
                'total_images': num_images,
                'total_detections': num_detections,
                'processing_time': f'{processing_time:.2f}s',
                'average_time_per_image': f'{processing_time/num_images:.3f}s',
                'mAP': f'{metrics["mAP"]:.4f}',
                'mean_precision': f'{metrics["mean_precision"]:.4f}',
                'mean_recall': f'{metrics["mean_recall"]:.4f}',
                'f1_score': f'{metrics["f1_score"]:.4f}'
            },
            'class_metrics': {}
        }
        
        for class_id, metric in metrics['class_metrics'].items():
            summary['class_metrics'][metric['class_name']] = {
                'AP': f'{metric["ap"]:.4f}',
                'Precision': f'{metric["precision"]:.4f}',
                'Recall': f'{metric["recall"]:.4f}',
                'True_Positives': metric['tp_count'],
                'False_Positives': metric['fp_count'],
                'False_Negatives': metric['fn_count']
            }
        
        # 保存JSON格式
        with open(self.metrics_dir / 'evaluation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 同时保存可读格式的报告
        with open(self.metrics_dir / 'evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("TRAFFIC SIGN DETECTION EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("SUMMARY:\n")
            f.write(f"  Total Images: {num_images}\n")
            f.write(f"  Total Detections: {num_detections}\n")
            f.write(f"  Processing Time: {processing_time:.2f}s\n")
            f.write(f"  Average Time per Image: {processing_time/num_images:.3f}s\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"  mAP: {metrics['mAP']:.4f}\n")
            f.write(f"  Mean Precision: {metrics['mean_precision']:.4f}\n")
            f.write(f"  Mean Recall: {metrics['mean_recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS:\n")
            for class_id, metric in metrics['class_metrics'].items():
                f.write(f"  {metric['class_name']}:\n")
                f.write(f"    AP: {metric['ap']:.4f}\n")
                f.write(f"    Precision: {metric['precision']:.4f}\n")
                f.write(f"    Recall: {metric['recall']:.4f}\n")
                f.write(f"    TP: {metric['tp_count']}, FP: {metric['fp_count']}, FN: {metric['fn_count']}\n\n")
    
    def debug_class_distribution(self, image_results: List[Dict]):
        """调试函数：检查数据中的类别分布"""
        gt_class_counts = defaultdict(int)
        pred_class_counts = defaultdict(int)
        
        print("=== 类别分布调试信息 ===")
        print(f"定义的类别数量: {len(self.class_names)} (ID: 0-{len(self.class_names)-1})")
        print(f"类别名称: {list(self.class_names.values()) if isinstance(self.class_names, dict) else self.class_names}")
        
        for result in image_results:
            # 统计真实标签
            for gt in result['ground_truths']:
                class_id = gt['class_id']
                gt_class_counts[class_id] += 1
                if class_id >= len(self.class_names):
                    print(f"警告: 发现超范围的真实类别ID: {class_id}")
            
            # 统计预测标签
            for pred in result['detections']:
                class_id = pred['class_id']
                pred_class_counts[class_id] += 1
                if class_id >= len(self.class_names):
                    print(f"警告: 发现超范围的预测类别ID: {class_id}")
        
        print(f"真实标签中的唯一类别ID: {sorted(gt_class_counts.keys())}")
        print(f"预测标签中的唯一类别ID: {sorted(pred_class_counts.keys())}")
        print(f"真实标签类别范围: {min(gt_class_counts.keys()) if gt_class_counts else 'N/A'} - {max(gt_class_counts.keys()) if gt_class_counts else 'N/A'}")
        print(f"预测标签类别范围: {min(pred_class_counts.keys()) if pred_class_counts else 'N/A'} - {max(pred_class_counts.keys()) if pred_class_counts else 'N/A'}")
        print("=== 调试信息结束 ===\n")
        
        return gt_class_counts, pred_class_counts

    def plot_confusion_matrix(self, image_results: List[Dict]):
        """绘制混淆矩阵"""
        y_true = []
        y_pred = []
        
        for result in image_results:
            # 收集真实标签和预测标签
            gt_classes = [gt['class_id'] for gt in result['ground_truths']]
            pred_classes = [pred['class_id'] for pred in result['detections']]
            
            # 简单匹配：每个真实标签对应最近的预测
            for gt_class in gt_classes:
                # 确保类别ID在有效范围内
                if gt_class >= len(self.class_names):
                    print(f"Warning: Found class_id {gt_class} which exceeds defined classes (max: {len(self.class_names)-1})")
                    continue
                    
                if pred_classes:
                    # 使用最高置信度的预测
                    best_pred = max(result['detections'], key=lambda x: x['combined_confidence'])
                    pred_class = best_pred['class_id']
                    
                    # 确保预测类别ID在有效范围内
                    if pred_class >= len(self.class_names):
                        print(f"Warning: Found predicted class_id {pred_class} which exceeds defined classes (max: {len(self.class_names)-1})")
                        continue
                        
                    y_true.append(gt_class)
                    y_pred.append(pred_class)
                else:
                    # 没有预测时跳过，不使用-1标记
                    continue
        
        if not y_true or not y_pred:
            return
        
        # 确保所有标签都在有效范围内
        valid_labels = list(range(len(self.class_names)))
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=valid_labels)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'confusion_matrix.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 生成分类报告
        try:
            # 确保标签数量匹配
            if len(set(y_true + y_pred)) <= len(self.class_names):
                report = classification_report(y_true, y_pred, 
                                             target_names=self.class_names,
                                             labels=valid_labels,
                                             output_dict=True, zero_division=0)
                
                # 保存分类报告
                with open(self.metrics_dir / 'classification_report.json', 'w') as f:
                    json.dump(report, f, indent=2)
            else:
                print("Warning: Skipping classification report due to label mismatch")
                
        except Exception as e:
            print(f"Warning: Could not generate classification report: {e}")
    
    def plot_detection_statistics(self, image_results: List[Dict]):
        """绘制检测统计图"""
        # 统计每张图片的检测数量
        detection_counts = [len(result['detections']) for result in image_results]
        gt_counts = [len(result['ground_truths']) for result in image_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 检测数量分布
        ax1.hist(detection_counts, bins=20, alpha=0.7, color='lightblue', label='Predictions')
        ax1.hist(gt_counts, bins=20, alpha=0.7, color='lightcoral', label='Ground Truth')
        ax1.set_xlabel('Number of Objects per Image')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Objects per Image Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 检测vs真实对比
        ax2.scatter(gt_counts, detection_counts, alpha=0.6)
        ax2.plot([0, max(max(gt_counts), max(detection_counts))], 
                [0, max(max(gt_counts), max(detection_counts))], 'r--', label='Perfect Match')
        ax2.set_xlabel('Ground Truth Count')
        ax2.set_ylabel('Prediction Count')
        ax2.set_title('Predictions vs Ground Truth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 类别分布（预测）
        pred_class_counts = defaultdict(int)
        for result in image_results:
            for detection in result['detections']:
                pred_class_counts[detection['class_id']] += 1
        
        if pred_class_counts:
            classes = [self.class_names[cid] for cid in pred_class_counts.keys()]
            counts = list(pred_class_counts.values())
            ax3.bar(classes, counts, alpha=0.7, color='skyblue')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Prediction Count')
            ax3.set_title('Predicted Class Distribution')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 类别分布（真实）
        gt_class_counts = defaultdict(int)
        for result in image_results:
            for gt in result['ground_truths']:
                gt_class_counts[gt['class_id']] += 1
        
        if gt_class_counts:
            classes = [self.class_names[cid] for cid in gt_class_counts.keys()]
            counts = list(gt_class_counts.values())
            ax4.bar(classes, counts, alpha=0.7, color='lightcoral')
            ax4.set_xlabel('Class')
            ax4.set_ylabel('Ground Truth Count')
            ax4.set_title('Ground Truth Class Distribution')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'detection_statistics.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()

    def save_augmentation_comparison(self, image_results: List[Dict]):
        """保存数据增强前后对比图"""
        if not image_results:
            return
        
        # 选择一些有检测结果的图像进行对比
        images_with_detections = [r for r in image_results if r['detections']]
        if not images_with_detections:
            return
        
        # 随机选择几张图片
        num_comparisons = min(4, len(images_with_detections))
        selected = random.sample(images_with_detections, num_comparisons)
        
        fig, axes = plt.subplots(num_comparisons, 2, figsize=(15, 4 * num_comparisons))
        if num_comparisons == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Data Augmentation Comparison\nLeft: Original, Right: Augmented with Detections', fontsize=16)
        
        for idx, result in enumerate(selected):
            # 原始图像
            original_image = cv2.imread(result['image_path'])
            if original_image is not None:
                original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                axes[idx, 0].imshow(original_rgb)
                axes[idx, 0].set_title(f'Original - {Path(result["image_path"]).name}')
                axes[idx, 0].axis('off')
            
            # 增强后的图像（带检测框）
            if 'augmented_image' in result:
                augmented_image = result['augmented_image'].copy()
                
                # 绘制检测框
                for detection in result['detections']:
                    bbox = detection['bbox']
                    class_name = detection['class_name']
                    confidence = detection['combined_confidence']
                    
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    cv2.rectangle(augmented_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label_text = f'{class_name}: {confidence:.2f}'
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(augmented_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                    cv2.putText(augmented_image, label_text, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                augmented_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                axes[idx, 1].imshow(augmented_rgb)
                axes[idx, 1].set_title(f'Augmented + Detections (Conf: {result["max_confidence"]:.3f})')
                axes[idx, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.examples_dir / 'augmentation_comparison.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()

    # ...existing code...
def apply_augmentation(image: np.ndarray) -> np.ndarray:
    """应用随机数据增强"""
    if not ENABLE_AUGMENTATION or random.random() > AUGMENTATION_PROBABILITY:
        return image
    
    # 随机选择增强类型
    augmentation_types = [
        # 天气效果
        lambda x: WeatherAugmentation.add_rain(x, random.uniform(0.1, 0.4)),
        lambda x: WeatherAugmentation.add_snow(x, random.uniform(0.1, 0.3)),
        lambda x: WeatherAugmentation.add_fog(x, random.uniform(0.2, 0.5)),
        
        # 模糊效果
        lambda x: WeatherAugmentation.add_blur(x, random.uniform(0.1, 0.4)),
        lambda x: WeatherAugmentation.add_motion_blur(x, random.randint(5, 20)),
        
        # 物理损坏
        lambda x: WeatherAugmentation.add_dirt_spots(x, random.randint(2, 8)),
        lambda x: WeatherAugmentation.add_scratches(x, random.randint(1, 4)),
        
        # 光照和对比度
        lambda x: WeatherAugmentation.adjust_brightness(x, random.uniform(0.6, 1.4)),
        lambda x: WeatherAugmentation.add_shadow(x, random.randint(1, 3)),
        
        # 畸变
        lambda x: WeatherAugmentation.simulate_lens_distortion(x, random.uniform(0.1, 0.3))
    ]
    
    # 随机应用1-2种增强（减少增强数量）
    num_augmentations = random.randint(1, 2)
    # 排除最耗时的增强操作
    fast_augmentation_types = [
        # 天气效果
        lambda x: WeatherAugmentation.add_rain(x, random.uniform(0.1, 0.4)),
        lambda x: WeatherAugmentation.add_snow(x, random.uniform(0.1, 0.3)),
        lambda x: WeatherAugmentation.add_fog(x, random.uniform(0.2, 0.5)),
        
        # 简单模糊效果
        lambda x: WeatherAugmentation.add_blur(x, random.uniform(0.1, 0.4)),
        
        # 物理损坏
        lambda x: WeatherAugmentation.add_dirt_spots(x, random.randint(2, 5)),
        lambda x: WeatherAugmentation.add_scratches(x, random.randint(1, 3)),
        
        # 光照
        lambda x: WeatherAugmentation.adjust_brightness(x, random.uniform(0.7, 1.3)),
    ]
    
    selected_augmentations = random.sample(fast_augmentation_types, 
                                         min(num_augmentations, len(fast_augmentation_types)))
    
    result = image.copy()
    for aug_func in selected_augmentations:
        try:
            result = aug_func(result)
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}")
            continue
    
    return result


def main():
    """主评估函数"""
    print("="*60)
    print("Traffic Sign Detection and Classification Model Evaluation")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # 检查模型文件
    print("Checking model files...")
    for model_path in [DETECT_MODEL_PATH, CLASSIFY_MODEL_PATH, LABELS_PATH]:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return
    
    # 加载数据集
    print(f"Loading dataset from: {DATASET_PATH}")
    try:
        dataset_loader = DatasetLoader(DATASET_PATH, EVAL_SPLIT)
        image_list = dataset_loader.get_image_list()
        print(f"Found {len(image_list)} images for evaluation")
        
        if len(image_list) == 0:
            print("Error: No images found in dataset")
            return
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 初始化推理系统
    print("Initializing inference system...")
    try:
        inference_system = TrafficSignInference(
            DETECT_MODEL_PATH,
            CLASSIFY_MODEL_PATH,
            LABELS_PATH
        )
        class_names = inference_system.classifier.class_names
        num_classes = inference_system.classifier.num_classes
        print(f"Loaded models with {num_classes} classes")
        
    except Exception as e:
        print(f"Error initializing inference system: {e}")
        return
    
    # 初始化评估指标和可视化器
    metrics_calculator = EvaluationMetrics(num_classes, class_names)
    visualizer = ResultVisualizer(OUTPUT_DIR, class_names)
    
    # 开始评估
    print(f"Starting evaluation on {len(image_list)} images...")
    print(f"Data augmentation: {'Enabled' if ENABLE_AUGMENTATION else 'Disabled'}")
    
    start_time = time.time()
    image_results = []
    total_detections = 0
    
    for i, image_path in enumerate(image_list):
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            
            img_height, img_width = image.shape[:2]
            
            # 应用数据增强
            augmented_image = apply_augmentation(image)
            
            # 使用内存临时文件或直接传递数组
            if hasattr(inference_system, 'process_image_array'):
                predictions = inference_system.process_image_array(augmented_image)
            else:
                # 使用内存缓冲区避免磁盘I/O
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as temp_file:
                    cv2.imwrite(temp_file.name, augmented_image)
                    predictions = inference_system.process_image(temp_file.name)
            
            # 加载真实标注
            ground_truths = dataset_loader.load_annotations(image_path)
            
            # 更新评估指标
            metrics_calculator.update(predictions, ground_truths, img_width, img_height)
            
            # 记录图像结果
            max_confidence = max([p['combined_confidence'] for p in predictions]) if predictions else 0.0
            image_results.append({
                'image_path': str(image_path),
                'augmented_image': augmented_image,  # 保存增强后的图像
                'detections': predictions,
                'ground_truths': ground_truths,
                'max_confidence': max_confidence
            })
            
            total_detections += len(predictions)
            
            # 显示进度（更频繁）
            if (i + 1) % 20 == 0 or (i + 1) == len(image_list):
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                eta = avg_time * (len(image_list) - i - 1)
                fps = (i + 1) / elapsed
                print(f"Progress: {i+1}/{len(image_list)} ({(i+1)/len(image_list)*100:.1f}%), "
                      f"Detections: {total_detections}, "
                      f"Time: {elapsed:.1f}s, ETA: {eta:.1f}s, "
                      f"Speed: {fps:.2f} img/s")
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    # 计算最终指标
    print("\nCalculating evaluation metrics...")
    final_metrics = metrics_calculator.calculate_map()
    
    # 打印结果
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total Images Processed: {len(image_results)}")
    print(f"Total Detections: {total_detections}")
    print(f"Processing Time: {processing_time:.2f}s")
    print(f"Average Time per Image: {processing_time/len(image_results):.3f}s")
    print("\nOverall Metrics:")
    print(f"  mAP: {final_metrics['mAP']:.4f}")
    print(f"  Mean Precision: {final_metrics['mean_precision']:.4f}")
    print(f"  Mean Recall: {final_metrics['mean_recall']:.4f}")
    print(f"  F1 Score: {final_metrics['f1_score']:.4f}")
    
    print("\nPer-Class Metrics:")
    for class_id, metric in final_metrics['class_metrics'].items():
        print(f"  {metric['class_name']}: "
              f"AP={metric['ap']:.3f}, "
              f"P={metric['precision']:.3f}, "
              f"R={metric['recall']:.3f}")
    
    # 生成可视化结果
    print("\nGenerating visualizations...")
    try:
        visualizer.save_example_images(image_results)
        visualizer.save_augmentation_comparison(image_results)
        visualizer.plot_confidence_distribution(image_results)
        visualizer.plot_class_performance(final_metrics)
        visualizer.plot_pr_curve(metrics_calculator)
        
        # 添加调试信息
        visualizer.debug_class_distribution(image_results)
        
        visualizer.plot_confusion_matrix(image_results)
        visualizer.plot_detection_statistics(image_results)
        visualizer.save_metrics_summary(final_metrics, processing_time, 
                                      len(image_results), total_detections)
        
        print(f"Evaluation completed! Results saved to: {OUTPUT_DIR}")
        print(f"  - Example images: {EXAMPLES_DIR}")
        print(f"  - Augmentation comparison: {EXAMPLES_DIR}/augmentation_comparison.png")
        print(f"  - Metrics charts: {METRICS_DIR}")
        print(f"  - Confusion matrix: {METRICS_DIR}/confusion_matrix.png")
        print(f"  - Detection statistics: {METRICS_DIR}/detection_statistics.png")
        print(f"  - Summary report: {METRICS_DIR}/evaluation_summary.json")
        print(f"  - Classification report: {METRICS_DIR}/classification_report.json")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    print("="*60)


if __name__ == "__main__":
    main()
