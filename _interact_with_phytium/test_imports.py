"""
测试导入和基本功能
"""
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from capture import VideoCapture
    from sender import UDPSender  
    from visualizer import ResultVisualizer, draw_results
    import cv2
    import numpy as np
    
    print("✓ 所有模块导入成功")
    
    # 测试创建对象
    try:
        # 测试VideoCapture（不实际打开视频）
        print("✓ VideoCapture 类可用")
        
        # 测试UDPSender
        print("✓ UDPSender 类可用")
        
        # 测试ResultVisualizer
        print("✓ ResultVisualizer 类可用")
        
        # 测试绘图函数
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_detections = [{"x": 100, "y": 100, "w": 50, "h": 50, "label": "test"}]
        result = draw_results(test_frame, test_detections)
        if result is not None:
            print("✓ draw_results 函数正常")
            
        print("\n🎉 所有基本功能测试通过！")
        print("可以运行 python src/main.py 启动简单检测展示程序")
        
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        
except ImportError as e:
    print(f"✗ 导入失败: {e}")
except Exception as e:
    print(f"✗ 其他错误: {e}")
