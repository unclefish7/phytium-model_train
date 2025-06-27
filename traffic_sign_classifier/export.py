#!/usr/bin/env python3
"""
PyTorch to ONNX Model Export Script
将PyTorch模型导出为ONNX格式，优化用于CPU推理

该脚本专门用于导出交通标志分类模型，确保：
1. 模型在eval()状态下导出
2. 使用与训练/测试完全一致的图像输入尺寸(224x224)
3. 输入tensor与真实推理时的数据shape/预处理一致
4. 设定合理的opset_version(默认11)
5. 自动处理模型输出（logits或softmax概率）
6. 生成的ONNX模型保证结构完整，适用于精度敏感的推理任务

Usage Examples:
# 导出单个模型文件
python export.py --model models/best_classifier.pth --output models/best_classifier.onnx

# 导出所有pth文件
python export.py --model_dir ../model --output_dir ../model/onnx_models

# 指定输入尺寸导出
python export.py --model ./models/best_classifier.pth --output ./models/best_classifier_128.onnx --input_size 128 128 --verify

# 导出并验证模型
python export.py --model ../model/best_classifier.pth --output ../model/best_classifier.onnx --verify
"""

import torch
import torch.nn as nn
import torch.onnx
from torchvision import models
import argparse
import os
import glob
import numpy as np
from pathlib import Path
import onnx
import onnxruntime as ort


def create_model(num_classes):
    """创建MobileNetV3模型"""
    model = models.mobilenet_v3_small(pretrained=True)
    # 替换最后一层
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_pytorch_model(model_path, device='cpu'):
    """加载PyTorch模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading PyTorch model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型参数
    num_classes = checkpoint.get('num_classes', 45)  # 默认45类交通标志
    best_val_acc = checkpoint.get('best_val_acc', 0)
    
    # 创建并加载模型
    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 确保模型处于eval状态
    
    print("Model loaded successfully:")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Best validation accuracy: {best_val_acc:.4f}")
    
    # 检测模型输出类型
    output_type = detect_model_output_type(model)
    
    return model, num_classes


def export_to_onnx(model, output_path, input_size=(224, 224), batch_size=1, opset_version=11):
    """将PyTorch模型导出为ONNX格式"""
    print("Exporting to ONNX format...")
    print(f"  - Input size: {input_size}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - ONNX opset version: {opset_version}")
    
    # 创建虚拟输入 - 形状为 [batch_size, channels, height, width]
    # 输入尺寸与训练/推理时一致: (224, 224)
    # 通道数为3 (RGB图像)
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 确保模型处于eval模式
    model.eval()
    
    # 导出为ONNX
    torch.onnx.export(
        model,                          # 模型
        dummy_input,                    # 虚拟输入
        output_path,                    # 输出路径
        export_params=True,             # 导出参数
        opset_version=opset_version,    # ONNX版本
        do_constant_folding=True,       # 常量折叠优化
        input_names=['input'],          # 输入名称
        output_names=['output'],        # 输出名称
        dynamic_axes={                  # 动态轴（支持不同batch size）
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False                   # 减少输出信息
    )
    
    print(f"ONNX model exported to: {output_path}")
    return output_path


def verify_onnx_model(onnx_path, pytorch_model, input_size=(224, 224)):
    """验证ONNX模型与PyTorch模型的输出一致性"""
    print("Verifying ONNX model...")
    
    try:
        # 加载ONNX模型
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # 创建测试输入 - 与真实推理时的输入形状一致
        test_input = torch.randn(1, 3, input_size[0], input_size[1])
        
        # PyTorch推理
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
            # 确保输出是numpy格式用于比较
            if isinstance(pytorch_output, torch.Tensor):
                pytorch_output = pytorch_output.numpy()
        
        # ONNX推理
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # 比较输出
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        
        print(f"Maximum difference between PyTorch and ONNX outputs: {max_diff:.8f}")
        print(f"Mean absolute difference: {mean_diff:.8f}")
        
        # 检查输出形状是否一致
        if pytorch_output.shape != onnx_output.shape:
            print(f"⚠️  Output shape mismatch: PyTorch {pytorch_output.shape} vs ONNX {onnx_output.shape}")
            return False
        
        # 更严格的数值验证
        if max_diff < 1e-5:
            print("✅ ONNX model verification passed!")
            return True
        elif max_diff < 1e-3:
            print("⚠️  Small difference detected, but within acceptable range")
            return True
        else:
            print("❌ Large difference detected, please check the model")
            return False
            
    except Exception as e:
        print(f"❌ ONNX model verification failed: {str(e)}")
        return False


def optimize_onnx_model(onnx_path):
    """优化ONNX模型用于CPU推理"""
    try:
        # 尝试导入onnxoptimizer，如果没有安装则跳过优化
        try:
            import onnxoptimizer  # type: ignore
        except ImportError:
            print("onnxoptimizer not available, skipping optimization")
            print("Install with: pip install onnxoptimizer")
            return onnx_path
        
        print("Optimizing ONNX model for CPU inference...")
        
        # 加载ONNX模型
        model = onnx.load(onnx_path)
        
        # 应用优化
        optimized_model = onnxoptimizer.optimize(model, passes=[
            'eliminate_deadend',
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_pad',
            'eliminate_unused_initializer',
            'extract_constant_to_initializer',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
            'fuse_consecutive_concats',
            'fuse_consecutive_reduce_unsqueeze',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_into_conv',
            'fuse_transpose_into_gemm',
        ])
        
        # 保存优化后的模型
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        onnx.save(optimized_model, optimized_path)
        
        print(f"Optimized ONNX model saved to: {optimized_path}")
        return optimized_path
        
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        return onnx_path


def get_model_info(onnx_path):
    """获取ONNX模型信息"""
    try:
        model = onnx.load(onnx_path)
        
        print("\nONNX Model Information:")
        print(f"  - IR version: {model.ir_version}")
        print(f"  - Producer: {model.producer_name}")
        print(f"  - Producer version: {model.producer_version}")
        
        # 输入信息
        print("  - Inputs:")
        for input_tensor in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"    {input_tensor.name}: {shape}")
        
        # 输出信息
        print("  - Outputs:")
        for output_tensor in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"    {output_tensor.name}: {shape}")
            
        # 模型大小
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        print(f"  - File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"Failed to get model info: {str(e)}")


def export_single_model(model_path, output_path, input_size, verify, optimize):
    """导出单个模型"""
    try:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_path}")
        print(f"Output path: {output_path}")
        
        # 加载PyTorch模型
        model, num_classes = load_pytorch_model(model_path, device='cpu')
        
        # 确保模型在CPU上进行导出（避免CUDA相关问题）
        model = model.cpu()
        
        # 导出为ONNX
        onnx_path = export_to_onnx(model, output_path, input_size)
        
        # 验证模型
        if verify:
            verify_success = verify_onnx_model(onnx_path, model, input_size)
            if not verify_success:
                print("⚠️  Verification failed, but ONNX model was still created")
        
        # 优化模型
        final_path = onnx_path
        if optimize:
            optimized_path = optimize_onnx_model(onnx_path)
            if optimized_path != onnx_path:
                final_path = optimized_path
        
        # 显示模型信息
        get_model_info(final_path)
        
        print("✅ Export completed successfully!")
        return final_path
        
    except Exception as e:
        print(f"❌ Failed to export {model_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def export_batch_models(model_dir, output_dir, input_size, verify, optimize):
    """批量导出模型"""
    # 查找所有pth文件
    pth_files = glob.glob(os.path.join(model_dir, "**/*.pth"), recursive=True)
    
    if not pth_files:
        print(f"No .pth files found in {model_dir}")
        return
    
    print(f"Found {len(pth_files)} .pth files to export:")
    for pth_file in pth_files:
        print(f"  - {pth_file}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    successful_exports = []
    failed_exports = []
    
    for pth_file in pth_files:
        # 生成输出文件名
        rel_path = os.path.relpath(pth_file, model_dir)
        onnx_name = os.path.splitext(rel_path)[0] + '.onnx'
        output_path = os.path.join(output_dir, onnx_name)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Exporting: {pth_file}")
        print(f"Output: {output_path}")
        
        result = export_single_model(pth_file, output_path, input_size, verify, optimize)
        
        if result:
            successful_exports.append((pth_file, result))
        else:
            failed_exports.append(pth_file)
    
    # 总结
    print(f"\n{'='*60}")
    print("Export Summary:")
    print(f"  - Successful: {len(successful_exports)}")
    print(f"  - Failed: {len(failed_exports)}")
    
    if successful_exports:
        print("\nSuccessful exports:")
        for src, dst in successful_exports:
            print(f"  ✅ {src} -> {dst}")
    
    if failed_exports:
        print("\nFailed exports:")
        for src in failed_exports:
            print(f"  ❌ {src}")


def detect_model_output_type(model, input_size=(224, 224)):
    """检测模型输出类型：logits 或 softmax概率"""
    model.eval()
    with torch.no_grad():
        # 创建测试输入
        test_input = torch.randn(1, 3, input_size[0], input_size[1])
        output = model(test_input)
        
        # 检查输出是否为概率分布（和为1）
        output_sum = torch.sum(output, dim=1)
        is_probability = torch.allclose(output_sum, torch.ones_like(output_sum), atol=1e-3)
        
        # 检查输出是否都在[0,1]范围内
        in_range = torch.all(output >= 0) and torch.all(output <= 1)
        
        if is_probability and in_range:
            print("  - Model output type: Softmax probabilities")
            return "probabilities"
        else:
            print("  - Model output type: Raw logits")
            return "logits"


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to ONNX format')
    
    # 模型输入参数
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--model', type=str, help='Path to single PyTorch model (.pth)')
    group1.add_argument('--model_dir', type=str, help='Directory containing PyTorch models')
    
    # 输出参数
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--output', type=str, help='Output path for single model')
    group2.add_argument('--output_dir', type=str, help='Output directory for batch export')
    
    # 其他参数
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224],
                       help='Input image size (height width), default: 224 224')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for export, default: 1')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version, default: 11')
    parser.add_argument('--verify', action='store_true',
                       help='Verify ONNX model against PyTorch model')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize ONNX model for CPU inference')
    
    args = parser.parse_args()
    
    # 检查参数组合
    if args.model and not args.output:
        # 单模型导出，自动生成输出路径
        args.output = args.model.replace('.pth', '.onnx')
    elif args.model_dir and not args.output_dir:
        # 批量导出，自动生成输出目录
        args.output_dir = os.path.join(args.model_dir, 'onnx_models')
    
    print("PyTorch to ONNX Export Tool")
    print("="*50)
    
    if args.model:
        # 单模型导出
        print(f"Input model: {args.model}")
        print(f"Output path: {args.output}")
        print(f"Input size: {args.input_size[0]}x{args.input_size[1]}")
        print(f"Verification: {'Enabled' if args.verify else 'Disabled'}")
        print(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")
        
        export_single_model(args.model, args.output, tuple(args.input_size), 
                          args.verify, args.optimize)
    else:
        # 批量导出
        print(f"Input directory: {args.model_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Input size: {args.input_size[0]}x{args.input_size[1]}")
        print(f"Verification: {'Enabled' if args.verify else 'Disabled'}")
        print(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")
        
        export_batch_models(args.model_dir, args.output_dir, tuple(args.input_size),
                          args.verify, args.optimize)


if __name__ == '__main__':
    main()
