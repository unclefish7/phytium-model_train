#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型可用性检查脚本
检查ONNX模型是否可用，以及输入输出形状
"""

# 使用方法：python src/check_availability.py .\models 

import os
import sys
import argparse
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np


def check_onnx_model(model_path):
    """
    检查ONNX模型的可用性和输入输出形状
    
    Args:
        model_path (str): ONNX模型文件路径
    
    Returns:
        dict: 包含模型信息的字典
    """
    result = {
        'model_path': model_path,
        'is_valid': False,
        'file_exists': False,
        'file_size_mb': 0,
        'onnx_version': None,
        'opset_version': None,
        'input_info': [],
        'output_info': [],
        'error_message': None
    }
    
    try:
        # 检查文件是否存在
        if not os.path.exists(model_path):
            result['error_message'] = f"模型文件不存在: {model_path}"
            return result
        
        result['file_exists'] = True
        result['file_size_mb'] = round(os.path.getsize(model_path) / (1024 * 1024), 2)
        
        print(f"正在检查模型: {model_path}")
        print(f"文件大小: {result['file_size_mb']} MB")
        
        # 加载ONNX模型
        model = onnx.load(model_path)
        
        # 检查模型是否有效
        onnx.checker.check_model(model)
        
        # 获取ONNX版本信息
        result['onnx_version'] = model.ir_version
        if model.opset_import:
            result['opset_version'] = model.opset_import[0].version
        
        # 获取输入信息
        for input_tensor in model.graph.input:
            input_info = {
                'name': input_tensor.name,
                'type': input_tensor.type.tensor_type.elem_type,
                'shape': []
            }
            
            # 获取输入形状
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    input_info['shape'].append(dim.dim_value)
                elif dim.dim_param:
                    input_info['shape'].append(dim.dim_param)
                else:
                    input_info['shape'].append('dynamic')
            
            result['input_info'].append(input_info)
        
        # 获取输出信息
        for output_tensor in model.graph.output:
            output_info = {
                'name': output_tensor.name,
                'type': output_tensor.type.tensor_type.elem_type,
                'shape': []
            }
            
            # 获取输出形状
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    output_info['shape'].append(dim.dim_value)
                elif dim.dim_param:
                    output_info['shape'].append(dim.dim_param)
                else:
                    output_info['shape'].append('dynamic')
            
            result['output_info'].append(output_info)
        
        # 尝试创建推理会话来验证模型是否可以运行
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            result['is_valid'] = True
            print("✓ 模型验证成功，可以正常加载和运行")
        except Exception as e:
            result['error_message'] = f"模型无法创建推理会话: {str(e)}"
            print(f"✗ 模型验证失败: {str(e)}")
        
    except onnx.checker.ValidationError as e:
        result['error_message'] = f"ONNX模型验证失败: {str(e)}"
        print(f"✗ ONNX模型验证失败: {str(e)}")
    except Exception as e:
        result['error_message'] = f"检查模型时发生错误: {str(e)}"
        print(f"✗ 检查模型时发生错误: {str(e)}")
    
    return result


def print_model_info(result):
    """
    打印模型信息
    
    Args:
        result (dict): check_onnx_model返回的结果字典
    """
    print("\n" + "="*60)
    print("模型检查结果")
    print("="*60)
    
    print(f"模型路径: {result['model_path']}")
    print(f"文件存在: {'是' if result['file_exists'] else '否'}")
    print(f"文件大小: {result['file_size_mb']} MB")
    print(f"模型有效: {'是' if result['is_valid'] else '否'}")
    
    if result['onnx_version']:
        print(f"ONNX IR版本: {result['onnx_version']}")
    if result['opset_version']:
        print(f"Opset版本: {result['opset_version']}")
    
    if result['error_message']:
        print(f"错误信息: {result['error_message']}")
    
    print("\n输入信息:")
    print("-" * 40)
    if result['input_info']:
        for i, input_info in enumerate(result['input_info']):
            shape_str = "x".join([str(s) for s in input_info['shape']])
            print(f"  输入 {i+1}: {input_info['name']}")
            print(f"    形状: [{shape_str}]")
            print(f"    类型: {input_info['type']}")
    else:
        print("  无输入信息")
    
    print("\n输出信息:")
    print("-" * 40)
    if result['output_info']:
        for i, output_info in enumerate(result['output_info']):
            shape_str = "x".join([str(s) for s in output_info['shape']])
            print(f"  输出 {i+1}: {output_info['name']}")
            print(f"    形状: [{shape_str}]")
            print(f"    类型: {output_info['type']}")
    else:
        print("  无输出信息")


def scan_directory_for_onnx(directory):
    """
    扫描目录中的所有ONNX文件
    
    Args:
        directory (str): 要扫描的目录路径
    
    Returns:
        list: ONNX文件路径列表
    """
    onnx_files = []
    directory = Path(directory)
    
    if directory.is_file() and directory.suffix.lower() == '.onnx':
        return [str(directory)]
    
    if directory.is_dir():
        for file_path in directory.rglob('*.onnx'):
            onnx_files.append(str(file_path))
    
    return onnx_files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检查ONNX模型可用性和输入输出形状')
    parser.add_argument('path', help='ONNX模型文件路径或包含ONNX文件的目录路径')
    parser.add_argument('--recursive', '-r', action='store_true', 
                       help='递归扫描目录中的所有ONNX文件')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='静默模式，减少输出信息')
    
    args = parser.parse_args()
    
    # 获取要检查的ONNX文件列表
    if os.path.isfile(args.path) and args.path.lower().endswith('.onnx'):
        onnx_files = [args.path]
    elif os.path.isdir(args.path):
        if args.recursive:
            onnx_files = scan_directory_for_onnx(args.path)
        else:
            onnx_files = [f for f in os.listdir(args.path) 
                         if f.lower().endswith('.onnx')]
            onnx_files = [os.path.join(args.path, f) for f in onnx_files]
    else:
        print(f"错误: 路径不存在或不是ONNX文件: {args.path}")
        sys.exit(1)
    
    if not onnx_files:
        print(f"在指定路径中未找到ONNX文件: {args.path}")
        sys.exit(1)
    
    print(f"找到 {len(onnx_files)} 个ONNX文件")
    
    # 检查每个ONNX文件
    results = []
    for onnx_file in onnx_files:
        if not args.quiet:
            print(f"\n{'='*80}")
        
        result = check_onnx_model(onnx_file)
        results.append(result)
        
        if not args.quiet:
            print_model_info(result)
    
    # 输出汇总信息
    valid_models = sum(1 for r in results if r['is_valid'])
    total_models = len(results)
    
    print(f"\n{'='*80}")
    print("检查汇总")
    print("="*80)
    print(f"总共检查: {total_models} 个模型")
    print(f"有效模型: {valid_models} 个")
    print(f"无效模型: {total_models - valid_models} 个")
    
    if valid_models < total_models:
        print("\n无效模型列表:")
        for result in results:
            if not result['is_valid']:
                print(f"  - {result['model_path']}: {result['error_message']}")


if __name__ == '__main__':
    main()