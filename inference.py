import cv2
import torch
import numpy as np
import os
import time
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v2
import onnxruntime

def preprocess_image(image_path, size=(320, 320)):
    """预处理图像用于模型输入"""
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 保存原始尺寸
    orig_h, orig_w = img.shape[:2]
    
    # 调整大小
    img_resized = cv2.resize(img, size)
    
    # 归一化
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # 转换为适合模型输入的格式 [1, C, H, W]
    img_input = img_normalized.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, axis=0)
    
    return img, img_input, (orig_h, orig_w)

def run_inference_pytorch(image_path, model, classes, confidence_threshold=0.5):
    """使用PyTorch模型进行推理"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 预处理图像
    orig_img, img_tensor, orig_size = preprocess_image(image_path)
    img_tensor = torch.tensor(img_tensor, dtype=torch.float32).to(device)
    
    # 记录推理时间
    start_time = time.time()
    
    # 前向传播
    with torch.no_grad():
        outputs = model([img_tensor])
    
    inference_time = time.time() - start_time
    
    # 处理输出
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    
    # 筛选置信度高的检测结果
    mask = scores >= confidence_threshold
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
    
    return orig_img, result_boxes, labels, scores, inference_time

def run_inference_onnx(image_path, onnx_path, classes, confidence_threshold=0.5):
    """使用ONNX模型进行推理"""
    # 创建ONNX运行时会话
    session = onnxruntime.InferenceSession(onnx_path)
    
    # 预处理图像
    orig_img, img_input, orig_size = preprocess_image(image_path)
    
    # 获取输入和输出名称
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # 记录推理时间
    start_time = time.time()
    
    # 执行推理
    outputs = session.run(output_names, {input_name: img_input})
    
    inference_time = time.time() - start_time
    
    # 解析输出 (具体处理方式取决于ONNX模型的输出格式)
    # 这里假设输出是边界框、分数和类别的列表
    boxes = outputs[0]
    scores = outputs[1]
    labels = outputs[2]
    
    # 筛选置信度高的检测结果
    mask = scores >= confidence_threshold
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
    
    return orig_img, result_boxes, labels, scores, inference_time

def visualize_results(image, boxes, labels, scores, classes, output_path=None):
    """可视化检测结果"""
    # 转换回BGR格式用于显示
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
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
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        class_name = classes[label - 1]  # 减1是因为索引从0开始，而类别从1开始
        label_text = f"{class_name}: {score:.2f}"
        
        # 确定标签的位置和大小
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        y1 = max(y1, text_height + 10)
        
        # 绘制标签背景和文本
        cv2.rectangle(
            image, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        cv2.putText(
            image, 
            label_text, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1
        )
    
    # 保存或显示结果
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"结果已保存到: {output_path}")
    else:
        cv2.imshow("Detection Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image

def main():
    # 设置参数
    model_path = os.path.join("models", "best_model.pth")
    onnx_path = os.path.join("models", "ssdlite_mobilenet_v2_quantized.onnx")
    test_image_path = os.path.join("data", "test_images", "test.jpg")
    output_path = os.path.join("results", "detection_result.jpg")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载类别信息
    with open(os.path.join("data", "processed_dataset", "classes.txt"), "r") as f:
        classes = [line.strip().split(": ")[1] for line in f.readlines()]
    
    num_classes = len(classes) + 1  # +1 用于背景类
    
    # 选择推理模式
    inference_mode = input("选择推理模式 (1: PyTorch, 2: ONNX): ")
    
    if inference_mode == "1":
        # 使用PyTorch模型
        model = ssdlite320_mobilenet_v2(pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        img, boxes, labels, scores, inference_time = run_inference_pytorch(
            test_image_path, model, classes
        )
    else:
        # 使用ONNX模型
        img, boxes, labels, scores, inference_time = run_inference_onnx(
            test_image_path, onnx_path, classes
        )
    
    # 可视化结果
    visualize_results(img, boxes, labels, scores, classes, output_path)
    
    print(f"推理完成！耗时: {inference_time*1000:.2f}ms")
    print(f"检测到{len(boxes)}个对象")

if __name__ == "__main__":
    main()
