"""
交通标志检测和分类推理系统

功能描述:
    结合检测模型和分类模型，对图像中的交通标志进行检测和分类
    - 检测模型: 检测交通标志的位置 (输入: 640x640, 输出: 25200个检测框)
    - 分类模型: 对检测到的交通标志进行分类 (输入: 224x224, 输出: 31个类别)

完整推理流程说明:

【第一阶段：目标检测与轻量级NMS】
1. 图像预处理：
   - 使用letterbox方法将输入图像缩放至640×640，保持长宽比
   - 图像归一化到[0,1]范围，转换为CHW格式

2. YOLOv5检测推理：
   - 输入：(1, 3, 640, 640)的图像tensor
   - 输出：(1, 25200, 6)的检测结果，每行包含[x_center, y_center, width, height, objectness, class_prob]
   - 注意：此检测模型仅提供目标定位能力，不包含具体类别分类

3. 第一阶段轻量级NMS过滤：
   - 步骤1：过滤objectness置信度低于OBJ_CONF_THRESHOLD(0.15)的检测框
   - 步骤2：将中心点坐标格式(cx,cy,w,h)转换为边界框格式(x1,y1,x2,y2)
   - 步骤3：在letterbox空间执行轻量级NMS，IoU阈值为IOU_THRESHOLD_STAGE1(0.95)
   - 步骤4：按objectness置信度排序，保留前MAX_DETECTIONS(100)个高质量候选框
   - 步骤5：将坐标从letterbox空间转换回原始图像空间
   - 目的：减少冗余检测框，防止低质量候选框污染后续分类器

【第二阶段：批量分类推理】
4. 候选框裁剪与预处理：
   - 根据检测框坐标从原图中裁剪出交通标志区域
   - 将所有裁剪区域统一resize到224×224尺寸（不保持长宽比）
   - 颜色空间转换：BGR → RGB
   - 图像归一化到[0,1]范围，转换为CHW格式
   - 组合成批处理tensor：shape为(N, 3, 224, 224)

5. 分类器批量推理：
   - 使用ONNX Runtime对所有候选框进行一次性批量分类
   - 输出：(N, 31)的logits矩阵，每行对应一个候选框的31类概率分布
   - 应用softmax激活函数获得归一化概率
   - 提取每个框的最高概率类别ID、置信度和类别名称

6. 结果整合：
   - 计算综合置信度：combined_confidence = detect_confidence × class_confidence
   - 构建完整检测结果，包含：bbox坐标、检测置信度、分类置信度、类别ID、类别名称

【第三阶段：分类NMS后处理】
7. 按类别分组NMS：
   - 将所有检测结果按class_id进行分组
   - 对每个类别组内的检测框：
     * 过滤综合置信度低于CONF_THRESHOLD_STAGE2(0.3)的结果
     * 执行类内NMS，IoU阈值为IOU_THRESHOLD_STAGE2(0.5)
     * 保留NMS后的最终检测结果
   - 目的：消除同类别内的重复检测，保证每个交通标志只有一个最佳检测框

【性能优化特点】
- 两阶段NMS设计：第一阶段快速过滤，第二阶段精确去重
- 批量分类推理：显著提升多目标场景下的推理速度
- 内存友好：通过MAX_DETECTIONS限制内存使用
- 坐标精确转换：保证检测框在不同空间的准确映射

使用方法:
    基本用法:
        python inference.py --input <图像文件夹路径> --output <输出文件夹路径>
    
    完整参数:
        python src/inference.py --input /workspace/dataset/images/test --output ./result --detect_model ./models/detect_only_LowRes.onnx --classify_model ./models/best_classifier.onnx --labels ./label.yaml

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
    2. 控制台输出: 实时处理进度和最终统计信息

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
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import yaml
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import time
import torch
import torchvision.ops

# 第一阶段轻量级筛选参数
OBJ_CONF_THRESHOLD = 0.15
IOU_THRESHOLD_STAGE1 = 0.95
MAX_DETECTIONS = 100

# 第二阶段NMS参数
CONF_THRESHOLD_STAGE2 = 0.3
IOU_THRESHOLD_STAGE2 = 0.5


class TrafficSignDetector:
    """交通标志检测器"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, nms_threshold: float = 0.8):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (640, 640)
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像 - 使用letterbox缩放保持纵横比"""
        # 保存原始尺寸
        self.original_shape = image.shape[:2]
        
        # 使用letterbox缩放保持纵横比
        resized, self.ratio, self.pad = self.letterbox(image, self.input_size)
        
        # 归一化到[0,1]并转换为float32
        normalized = resized.astype(np.float32) / 255.0
        
        # 转换为CHW格式并添加batch维度
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def letterbox(self, image: np.ndarray, new_shape=(640, 640), color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True, stride=32):
        """使用letterbox缩放图像，保持纵横比 - 基于YOLOv5的letterbox函数"""
        shape = image.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return image, ratio, (dw, dh)
    
    def postprocess(self, output: np.ndarray) -> List[Dict]:
        """后处理检测结果 - 基于YOLOv5的处理方式，加入第一阶段轻量级NMS"""
        # output shape: [1, 25200, 6] -> [x_center, y_center, width, height, objectness, class_prob]
        # 注意：此检测模型只有检测能力，没有分类能力，所以只使用objectness作为置信度
        detections = output[0]  # Remove batch dimension [25200, 6]
        
        # 获取objectness置信度（第5列是objectness，忽略第6列的类别概率）
        objectness = detections[:, 4]  # objectness置信度
        
        # 第一步：过滤低置信度检测
        conf_mask = objectness >= OBJ_CONF_THRESHOLD
        if not np.any(conf_mask):
            return []
        
        filtered_detections = detections[conf_mask]
        filtered_confidences = objectness[conf_mask]
        
        # 第二步：转换坐标格式 (center_x, center_y, w, h) -> (x1, y1, x2, y2)
        boxes = np.copy(filtered_detections[:, :4])
        boxes[:, 0] = filtered_detections[:, 0] - filtered_detections[:, 2] / 2  # x1
        boxes[:, 1] = filtered_detections[:, 1] - filtered_detections[:, 3] / 2  # y1
        boxes[:, 2] = filtered_detections[:, 0] + filtered_detections[:, 2] / 2  # x2
        boxes[:, 3] = filtered_detections[:, 1] + filtered_detections[:, 3] / 2  # y2
        
        # 第三步：第一阶段轻量级NMS（在letterbox空间进行）
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(filtered_confidences, dtype=torch.float32)
        
        # 执行第一阶段NMS
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, IOU_THRESHOLD_STAGE1)
        
        # 保留NMS后的结果
        nms_boxes = boxes[keep_indices.cpu().numpy()]
        nms_confidences = filtered_confidences[keep_indices.cpu().numpy()]
        
        # 第四步：按置信度排序，保留前MAX_DETECTIONS个
        if len(nms_confidences) > MAX_DETECTIONS:
            top_indices = np.argsort(nms_confidences)[-MAX_DETECTIONS:]
            nms_boxes = nms_boxes[top_indices]
            nms_confidences = nms_confidences[top_indices]
        
        # 第五步：将坐标从letterbox空间转换回原始图像空间
        nms_boxes = self.scale_coords(self.input_size, nms_boxes, self.original_shape)
        
        results = []
        for i in range(len(nms_boxes)):
            x1, y1, x2, y2 = nms_boxes[i].astype(int)
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, self.original_shape[1] - 1))
            y1 = max(0, min(y1, self.original_shape[0] - 1))
            x2 = max(0, min(x2, self.original_shape[1] - 1))
            y2 = max(0, min(y2, self.original_shape[0] - 1))
            
            # 检测模型只提供检测能力，类别设为0（通用目标）
            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(nms_confidences[i]),
                'class': 0  # 检测模型没有分类能力，统一设为0
            })
        
        return results
    
    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        """将坐标从letterbox图像缩放回原始图像 - 基于YOLOv5的scale_boxes函数"""
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
        return coords
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测交通标志"""
        input_tensor = self.preprocess(image)
        output = self.session.run([self.output_name], {self.input_name: input_tensor})
        return self.postprocess(output[0])


class TrafficSignClassifier:
    """交通标志分类器"""
    
    def __init__(self, model_path: str, label_path: str):
        self.model_path = model_path
        self.input_size = (224, 224)
        
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
        """预处理图像 - 直接resize到224x224，不保持宽高比"""
        # 直接resize到模型输入尺寸，不保持宽高比
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # 转换BGR到RGB (如果训练时使用RGB)
        # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0,1]并转换为float32
        normalized = resized.astype(np.float32) / 255.0
        
        # 转换为CHW格式并添加batch维度
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def classify_batch(self, images: List[np.ndarray]) -> List[Tuple[int, float]]:
        """批量分类交通标志 - 优化版本"""
        if not images:
            return []
        
        # 预处理所有图像并组成batch
        batch_images = []
        for image in images:
            # 直接resize到模型输入尺寸，不保持宽高比
            resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
            
            # 转换BGR到RGB
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # 归一化到[0,1]并转换为float32
            normalized = resized.astype(np.float32) / 255.0
            
            # 转换为CHW格式
            input_tensor = np.transpose(normalized, (2, 0, 1))
            batch_images.append(input_tensor)
        
        # 组成batch - shape: (N, 3, 224, 224)
        batch_tensor = np.array(batch_images)
        
        # 批量推理
        output = self.session.run([self.output_name], {self.input_name: batch_tensor})
        predictions = output[0]  # shape: [N, num_classes]
        
        # 处理每个预测结果
        results = []
        for pred in predictions:
            # 应用softmax激活函数
            softmax_pred = self.softmax(pred)
            class_id = np.argmax(softmax_pred)
            confidence = float(softmax_pred[class_id])
            results.append((class_id, confidence))
        
        return results
    
    def softmax(self, x):
        """softmax激活函数"""
        exp_x = np.exp(x - np.max(x))  # 减去最大值防止溢出
        return exp_x / np.sum(exp_x)


class TrafficSignInference:
    """交通标志检测和分类推理系统"""
    
    def __init__(self, detect_model_path: str, classify_model_path: str, label_path: str):
        self.detector = TrafficSignDetector(detect_model_path)
        self.classifier = TrafficSignClassifier(classify_model_path, label_path)
        
    def process_image(self, image_path: str) -> List[Dict]:
        """处理单张图像 - 优化版本"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        # 第一阶段：检测交通标志（已包含轻量级NMS）
        detections = self.detector.detect(image)
        
        if not detections:
            return []
        
        # 第二阶段：批量裁剪检测区域并预处理
        crops = []
        valid_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # 确保坐标有效
            if x2 <= x1 or y2 <= y1:
                continue
                
            # 裁剪检测区域
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            crops.append(crop)
            valid_detections.append(detection)
        
        if not crops:
            return []
        
        # 第三阶段：批量分类
        classification_results = self.classifier.classify_batch(crops)
        
        # 第四阶段：组合结果
        results = []
        for detection, (class_id, class_conf) in zip(valid_detections, classification_results):
            class_name = self.classifier.class_names[class_id]
            
            results.append({
                'bbox': detection['bbox'],
                'detect_confidence': detection['confidence'],
                'class_id': class_id,
                'class_name': class_name,
                'class_confidence': class_conf,
                'combined_confidence': detection['confidence'] * class_conf
            })
        
        # 第五阶段：第二阶段分类NMS（按类别分组后进行NMS）
        results = self.apply_class_wise_nms(results)
        
        return results
    
    def apply_class_wise_nms(self, detections: List[Dict]) -> List[Dict]:
        """对分类后的结果进行按类别的NMS"""
        if not detections:
            return []
        
        # 按类别分组
        class_groups = {}
        for det in detections:
            class_id = det['class_id']
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(det)
        
        final_results = []
        
        # 对每个类别分别进行NMS
        for class_id, group_detections in class_groups.items():
            if not group_detections:
                continue
            
            # 过滤低置信度结果
            filtered_detections = [det for det in group_detections 
                                 if det['combined_confidence'] >= CONF_THRESHOLD_STAGE2]
            
            if not filtered_detections:
                continue
            
            # 准备NMS所需的tensor
            boxes = torch.tensor([det['bbox'] for det in filtered_detections], dtype=torch.float32)
            scores = torch.tensor([det['combined_confidence'] for det in filtered_detections], dtype=torch.float32)
            
            # 执行NMS
            keep_indices = torchvision.ops.nms(boxes, scores, IOU_THRESHOLD_STAGE2)
            
            # 保留NMS后的结果
            for idx in keep_indices:
                final_results.append(filtered_detections[idx])
        
        return final_results
    
    def process_folder(self, input_folder: str, output_folder: str = None):
        """处理文件夹中的所有图像"""
        input_path = Path(input_folder)
        if not input_path.exists():
            raise ValueError(f"Input folder does not exist: {input_folder}")
        
        # 创建输出文件夹
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in folder: {input_folder}")
            return
        
        print(f"Found {len(image_files)} images")
        
        total_detections = 0
        processed_images = 0
        processing_times = []
        
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            start_time = time.time()
            try:
                results = self.process_image(str(image_file))
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                processed_images += 1
                total_detections += len(results)
                
                # 保存可视化结果
                if output_folder:
                    self.visualize_results(str(image_file), results, 
                                         str(output_path / f"result_{image_file.name}"))
                
                print(f"  Detected {len(results)} traffic signs (Time: {processing_time:.3f}s)")
                
            except Exception as e:
                print(f"  Processing failed: {e}")
                continue
        
        # 打印统计信息
        print("\n" + "="*50)
        print("Processing completed! Statistics:")
        print(f"Total images: {len(image_files)}")
        print(f"Successfully processed: {processed_images}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections / processed_images if processed_images > 0 else 0:.2f}")
        print(f"Average processing time: {np.mean(processing_times) if processing_times else 0:.3f}s")
        print(f"Total processing time: {sum(processing_times):.3f}s")
        print("="*50)
    
    def visualize_results(self, image_path: str, detections: List[Dict], output_path: str):
        """可视化检测结果"""
        image = cv2.imread(image_path)
        
        # 绘制检测框和标签
        for detection in detections:
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
        raise FileNotFoundError(f"Detection model file not found: {args.detect_model}")
    if not os.path.exists(args.classify_model):
        raise FileNotFoundError(f"Classification model file not found: {args.classify_model}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Label file not found: {args.labels}")
    
    # 创建推理系统
    inference_system = TrafficSignInference(
        args.detect_model, 
        args.classify_model, 
        args.labels
    )
    
    # 处理图像
    print("Starting image processing...")
    inference_system.process_folder(args.input, args.output)


if __name__ == "__main__":
    main()