"""
交通标志检测和分类推理系统

功能描述:
    结合检测模型和分类模型，对图像中的交通标志进行检测和分类
    - 检测模型: 检测交通标志的位置 (输入: 640x640, 输出: 25200个检测框)
    - 分类模型: 对检测到的交通标志进行分类 (输入: 128x128, 输出: 31个类别)

使用方法:
    基本用法:
        python inference.py --input <图像文件夹路径> --output <输出文件夹路径>
    
    完整参数:
        python _detect_and_classify\src\inference.py --input E:\tt100k\phytium-model_train\yolo_dataset\yolo_dataset_one_cls\images\test --output _detect_and_classify\result --detect_model _detect_and_classify\models\detect_only_LowRes.onnx --classify_model _detect_and_classify\models\best_classifier.onnx --labels _detect_and_classify\label.yaml
        
        python inference.py \
            --input /path/to/images \
            --output /path/to/results \
            --detect_model ../models/detect_only_LowRes.onnx \
            --classify_model ../models/best_classifier.onnx \
            --labels ../label.yaml

参数说明:
    --input, -i     : 输入图像文件夹路径 (必需)
    --output, -o    : 输出结果文件夹路径 (可选，不指定则不保存可视化结果)
    --detect_model  : 检测模型路径 (默认: ../models/detect_only_LowRes.onnx)
    --classify_model: 分类模型路径 (默认: ../models/best_classifier.onnx)
    --labels        : 标签文件路径 (默认: ../label.yaml)

输出内容:
    1. 可视化图像: 每张输入图像对应一张带有检测框和分类标签的结果图像
    2. JSON结果文件: inference_results.json，包含详细的检测和分类结果
    3. 控制台输出: 实时处理进度和最终统计信息

支持的图像格式:
    .jpg, .jpeg, .png, .bmp, .tiff, .tif

示例:
    # 处理单个文件夹的图像
    python inference.py -i ./test_images -o ./results
    
    # 只进行推理不保存可视化结果
    python inference.py -i ./test_images
    
    # 使用自定义模型路径
    python inference.py -i ./test_images -o ./results \
        --detect_model ./models/my_detector.onnx \
        --classify_model ./models/my_classifier.onnx

输出JSON格式:
    {
        "statistics": {
            "total_images": 总图像数,
            "processed_images": 成功处理的图像数,
            "total_detections": 总检测数,
            "avg_detections_per_image": 平均每张图像检测数,
            "avg_processing_time": 平均处理时间(秒),
            "total_processing_time": 总处理时间(秒)
        },
        "results": {
            "image_name.jpg": {
                "image_path": "图像路径",
                "image_shape": [高度, 宽度, 通道数],
                "num_detections": 检测数量,
                "detections": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "detect_confidence": 检测置信度,
                        "class_id": 类别ID,
                        "class_name": "类别名称",
                        "class_confidence": 分类置信度,
                        "combined_confidence": 综合置信度
                    }
                ]
            }
        }
    }
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import yaml
from pathlib import Path
import argparse
import json
from typing import List, Tuple, Dict
import time


class TrafficSignDetector:
    """交通标志检测器"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, nms_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (640, 640)
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 保存原始尺寸
        self.original_shape = image.shape[:2]
        
        # Resize到模型输入尺寸
        resized = cv2.resize(image, self.input_size)
        
        # 归一化到[0,1]
        normalized = resized.astype(np.float32) / 255.0
        
        # 转换为CHW格式并添加batch维度
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess(self, output: np.ndarray) -> List[Dict]:
        """后处理检测结果"""
        # output shape: [1, 25200, 6] -> [x_center, y_center, width, height, confidence, class]
        detections = output[0]  # Remove batch dimension
        
        # 过滤低置信度检测
        conf_mask = detections[:, 4] >= self.conf_threshold
        detections = detections[conf_mask]
        
        if len(detections) == 0:
            return []
        
        # 转换坐标格式 (center_x, center_y, w, h) -> (x1, y1, x2, y2)
        boxes = np.copy(detections[:, :4])
        boxes[:, 0] = detections[:, 0] - detections[:, 2] / 2  # x1
        boxes[:, 1] = detections[:, 1] - detections[:, 3] / 2  # y1
        boxes[:, 2] = detections[:, 0] + detections[:, 2] / 2  # x2
        boxes[:, 3] = detections[:, 1] + detections[:, 3] / 2  # y2
        
        # 转换坐标到原始图像尺寸
        scale_x = self.original_shape[1] / self.input_size[0]
        scale_y = self.original_shape[0] / self.input_size[1]
        
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        # NMS
        confidences = detections[:, 4]
        classes = detections[:, 5].astype(int)
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            confidences.tolist(), 
            self.conf_threshold, 
            self.nms_threshold
        )
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i].astype(int)
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, self.original_shape[1] - 1))
                y1 = max(0, min(y1, self.original_shape[0] - 1))
                x2 = max(0, min(x2, self.original_shape[1] - 1))
                y2 = max(0, min(y2, self.original_shape[0] - 1))
                
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidences[i]),
                    'class': int(classes[i])
                })
        
        return results
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测交通标志"""
        input_tensor = self.preprocess(image)
        output = self.session.run([self.output_name], {self.input_name: input_tensor})
        return self.postprocess(output[0])


class TrafficSignClassifier:
    """交通标志分类器"""
    
    def __init__(self, model_path: str, label_path: str):
        self.model_path = model_path
        self.input_size = (128, 128)
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 加载标签
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = yaml.safe_load(f)
        self.class_names = label_data['names']
        self.num_classes = label_data['nc']
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # Resize到模型输入尺寸
        resized = cv2.resize(image, self.input_size)
        
        # 归一化到[0,1]
        normalized = resized.astype(np.float32) / 255.0
        
        # 转换为CHW格式并添加batch维度
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def classify(self, image: np.ndarray) -> Tuple[int, float]:
        """分类交通标志"""
        input_tensor = self.preprocess(image)
        output = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # 获取预测结果
        predictions = output[0][0]  # Remove batch dimension
        class_id = np.argmax(predictions)
        confidence = float(np.max(predictions))
        
        return class_id, confidence


class TrafficSignInference:
    """交通标志检测和分类推理系统"""
    
    def __init__(self, detect_model_path: str, classify_model_path: str, label_path: str):
        self.detector = TrafficSignDetector(detect_model_path)
        self.classifier = TrafficSignClassifier(classify_model_path, label_path)
        
    def process_image(self, image_path: str) -> Dict:
        """处理单张图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 检测交通标志
        detections = self.detector.detect(image)
        
        # 对每个检测结果进行分类
        results = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # 裁剪检测区域
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            # 分类
            class_id, class_conf = self.classifier.classify(crop)
            class_name = self.classifier.class_names[class_id]
            
            results.append({
                'bbox': detection['bbox'],
                'detect_confidence': detection['confidence'],
                'class_id': class_id,
                'class_name': class_name,
                'class_confidence': class_conf,
                'combined_confidence': detection['confidence'] * class_conf
            })
        
        return {
            'image_path': image_path,
            'image_shape': image.shape,
            'detections': results,
            'num_detections': len(results)
        }
    
    def process_folder(self, input_folder: str, output_folder: str = None) -> Dict:
        """处理文件夹中的所有图像"""
        input_path = Path(input_folder)
        if not input_path.exists():
            raise ValueError(f"输入文件夹不存在: {input_folder}")
        
        # 创建输出文件夹
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"在文件夹 {input_folder} 中未找到图像文件")
            return {}
        
        print(f"找到 {len(image_files)} 张图像")
        
        all_results = {}
        total_detections = 0
        processing_times = []
        
        for i, image_file in enumerate(image_files):
            print(f"处理 {i+1}/{len(image_files)}: {image_file.name}")
            
            start_time = time.time()
            try:
                result = self.process_image(str(image_file))
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                all_results[image_file.name] = result
                total_detections += result['num_detections']
                
                # 保存可视化结果
                if output_folder:
                    self.visualize_results(str(image_file), result, 
                                         str(output_path / f"result_{image_file.name}"))
                
                print(f"  检测到 {result['num_detections']} 个交通标志 (耗时: {processing_time:.3f}s)")
                
            except Exception as e:
                print(f"  处理失败: {e}")
                continue
        
        # 统计信息
        stats = {
            'total_images': len(image_files),
            'processed_images': len(all_results),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(all_results) if all_results else 0,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'total_processing_time': sum(processing_times)
        }
        
        # 保存结果
        if output_folder:
            results_file = output_path / "inference_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'statistics': stats,
                    'results': all_results
                }, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {results_file}")
        
        return {
            'statistics': stats,
            'results': all_results
        }
    
    def visualize_results(self, image_path: str, result: Dict, output_path: str):
        """可视化检测结果"""
        image = cv2.imread(image_path)
        
        # 绘制检测框和标签
        for detection in result['detections']:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            detect_conf = detection['detect_confidence']
            class_conf = detection['class_confidence']
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {detect_conf:.2f}|{class_conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 绘制标签背景
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # 绘制标签文字
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 保存结果图像
        cv2.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser(description='交通标志检测和分类推理')
    parser.add_argument('--input', '-i', required=True, help='输入图像文件夹路径')
    parser.add_argument('--output', '-o', help='输出结果文件夹路径')
    parser.add_argument('--detect_model', default='../models/detect_only_LowRes.onnx', 
                       help='检测模型路径')
    parser.add_argument('--classify_model', default='../models/best_classifier.onnx', 
                       help='分类模型路径')
    parser.add_argument('--labels', default='../label.yaml', help='标签文件路径')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.detect_model):
        raise FileNotFoundError(f"检测模型文件不存在: {args.detect_model}")
    if not os.path.exists(args.classify_model):
        raise FileNotFoundError(f"分类模型文件不存在: {args.classify_model}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"标签文件不存在: {args.labels}")
    
    # 创建推理系统
    inference_system = TrafficSignInference(
        args.detect_model, 
        args.classify_model, 
        args.labels
    )
    
    # 处理图像
    print("开始处理图像...")
    results = inference_system.process_folder(args.input, args.output)
    
    # 打印统计信息
    stats = results['statistics']
    print("\n" + "="*50)
    print("处理完成！统计信息:")
    print(f"总图像数: {stats['total_images']}")
    print(f"成功处理: {stats['processed_images']}")
    print(f"总检测数: {stats['total_detections']}")
    print(f"平均每张图像检测数: {stats['avg_detections_per_image']:.2f}")
    print(f"平均处理时间: {stats['avg_processing_time']:.3f}s")
    print(f"总处理时间: {stats['total_processing_time']:.3f}s")
    print("="*50)


if __name__ == "__main__":
    main()