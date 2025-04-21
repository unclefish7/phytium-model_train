import torch
import torchvision
import onnx
import onnxruntime
import numpy as np
import os
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v2

def prepare_input(batch_size=1):
    """准备用于ONNX导出的输入张量"""
    # SSD使用的输入大小为320x320
    x = torch.randn(batch_size, 3, 320, 320, requires_grad=True)
    return x

def convert_to_onnx():
    # 设置路径
    model_path = os.path.join("models", "best_model.pth")
    output_path = os.path.join("models", "ssdlite_mobilenet_v2.onnx")
    
    # 加载类别信息
    with open(os.path.join("data", "processed_dataset", "classes.txt"), "r") as f:
        classes = [line.strip().split(": ")[1] for line in f.readlines()]
    
    num_classes = len(classes) + 1  # +1 用于背景类
    
    # 加载模型
    model = ssdlite320_mobilenet_v2(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 准备输入张量
    x = prepare_input()
    
    # 导出为ONNX格式
    torch.onnx.export(model, 
                     x,
                     output_path,
                     export_params=True,
                     opset_version=12,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['boxes', 'scores', 'classes'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                  'boxes': {0: 'batch_size'},
                                  'scores': {0: 'batch_size'},
                                  'classes': {0: 'batch_size'}})
    
    print(f"模型已导出为ONNX格式: {output_path}")
    
    # 验证ONNX模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过！")
    
    # 进一步优化ONNX模型
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantized_output_path = os.path.join("models", "ssdlite_mobilenet_v2_quantized.onnx")
    
    quantize_dynamic(
        output_path,
        quantized_output_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f"量化后的ONNX模型已保存: {quantized_output_path}")
    
    # 比较模型大小
    original_size = os.path.getsize(output_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_output_path) / (1024 * 1024)
    
    print(f"原始ONNX模型大小: {original_size:.2f} MB")
    print(f"量化后ONNX模型大小: {quantized_size:.2f} MB")
    print(f"模型大小减少了: {(original_size - quantized_size) / original_size * 100:.2f}%")

if __name__ == "__main__":
    convert_to_onnx()
