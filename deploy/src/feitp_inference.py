import cv2
import numpy as np
import os
import time
import onnxruntime

class TrafficSignDetector:
    def __init__(self, model_path, classes_path, confidence_threshold=0.5):
        """
        初始化交通标志检测器
        Args:
            model_path: ONNX模型路径
            classes_path: 类别文件路径
            confidence_threshold: 置信度阈值
        """
        # 加载ONNX模型
        self.session = onnxruntime.InferenceSession(model_path)
        
        # 获取输入和输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 设置置信度阈值
        self.confidence_threshold = confidence_threshold
        
        # 加载类别
        with open(classes_path, "r") as f:
            self.classes = [line.strip().split(": ")[1] for line in f.readlines()]
        
        print(f"模型加载完成，共{len(self.classes)}个类别")
    
    def preprocess(self, image):
        """
        预处理图像
        Args:
            image: 输入图像
        Returns:
            处理后的图像和原始尺寸
        """
        # 保存原始尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 调整大小到320x320
        img_resized = cv2.resize(image, (320, 320))
        
        # 转换为RGB
        if len(img_resized.shape) == 2:  # 灰度图像
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 3:  # BGR图像
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_normalized = (img_normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # 转换为适合模型输入的格式 [1, C, H, W]
        img_input = img_normalized.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0)
        
        return img_input, (orig_h, orig_w)
    
    def detect(self, image):
        """
        检测图像中的交通标志
        Args:
            image: 输入图像(BGR格式)
        Returns:
            检测结果和推理时间
        """
        # 预处理图像
        img_input, orig_size = self.preprocess(image)
        
        # 记录推理时间
        start_time = time.time()
        
        # 执行推理
        outputs = self.session.run(self.output_names, {self.input_name: img_input})
        
        inference_time = time.time() - start_time
        
        # 解析输出
        boxes = outputs[0]
        scores = outputs[1]
        labels = outputs[2]
        
        # 筛选置信度高的检测结果
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        # 将框的坐标转换回原始图像大小
        orig_h, orig_w = orig_size
        scale_x, scale_y = orig_w / 320, orig_h / 320
        
        result_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            result_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return {
            'boxes': result_boxes,
            'labels': labels,
            'scores': scores,
            'inference_time': inference_time
        }
    
    def visualize(self, image, results, output_path=None):
        """
        可视化检测结果
        Args:
            image: 原始图像
            results: 检测结果
            output_path: 输出路径
        Returns:
            带有检测框的图像
        """
        # 获取检测结果
        boxes = results['boxes']
        labels = results['labels']
        scores = results['scores']
        
        # 复制图像以免修改原始图像
        vis_image = image.copy()
        
        # 为不同类别生成不同颜色
        colors = {}
        for label in set(labels):
            # 为每个类别随机生成一个颜色
            colors[label] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
        
        # 绘制边界框和类别标签
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            color = colors[label]
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            class_name = self.classes[label - 1]  # 减1是因为索引从0开始，而类别从1开始
            label_text = f"{class_name}: {score:.2f}"
            
            # 确定标签的位置和大小
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            y1 = max(y1, text_height + 10)
            
            # 绘制标签背景和文本
            cv2.rectangle(
                vis_image, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            cv2.putText(
                vis_image, 
                label_text, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                1
            )
        
        # 添加推理时间信息
        inference_time = results['inference_time']
        cv2.putText(
            vis_image,
            f"Inference: {inference_time*1000:.1f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
        # 保存或显示结果
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"结果已保存到: {output_path}")
        
        return vis_image

def process_video(video_path, detector, output_path=None, display=True):
    """
    处理视频文件
    Args:
        video_path: 视频文件路径
        detector: 交通标志检测器
        output_path: 输出视频路径
        display: 是否显示处理过程
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    avg_inference_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测交通标志
        results = detector.detect(frame)
        avg_inference_time += results['inference_time']
        
        # 可视化结果
        vis_frame = detector.visualize(frame, results)
        
        # 显示进度
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"处理帧: {frame_idx}/{frame_count}, 平均推理时间: {(avg_inference_time*1000/frame_idx):.1f}ms")
        
        # 保存到输出视频
        if output_path:
            out.write(vis_frame)
        
        # 显示
        if display:
            cv2.imshow('Traffic Sign Detection', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 清理
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"视频处理完成! 平均推理时间: {(avg_inference_time*1000/frame_idx):.1f}ms")

def main():
    # 设置参数
    model_path = "../model/ssdlite_mobilenet_v2_quantized.onnx"
    classes_path = "../data/classes.txt"
    test_image_path = "../data/test.jpg"
    test_video_path = "../data/test.mp4"
    output_dir = "../results"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化检测器
    detector = TrafficSignDetector(model_path, classes_path)
    
    # 选择模式
    mode = input("选择模式 (1: 图像, 2: 视频, 3: 摄像头): ")
    
    if mode == "1" and os.path.exists(test_image_path):
        # 处理单张图像
        image = cv2.imread(test_image_path)
        results = detector.detect(image)
        vis_image = detector.visualize(image, results, os.path.join(output_dir, "result.jpg"))
        
        cv2.imshow('Traffic Sign Detection', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif mode == "2" and os.path.exists(test_video_path):
        # 处理视频
        process_video(
            test_video_path,
            detector,
            os.path.join(output_dir, "result_video.mp4"),
            display=True
        )
    
    elif mode == "3":
        # 使用摄像头
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测交通标志
            results = detector.detect(frame)
            
            # 可视化结果
            vis_frame = detector.visualize(frame, results)
            
            # 显示
            cv2.imshow('Traffic Sign Detection', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("无效的模式选择或文件不存在")

if __name__ == "__main__":
    main()
