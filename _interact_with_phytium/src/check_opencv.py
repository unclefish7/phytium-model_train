#!/usr/bin/env python3
"""
检查OpenCV版本和追踪器可用性
"""
import cv2
import numpy as np

def check_opencv_info():
    """检查OpenCV基本信息"""
    print(f"OpenCV版本: {cv2.__version__}")
    print("OpenCV构建信息:")
    print(cv2.getBuildInformation())
    print("\n" + "="*50 + "\n")

def test_trackers():
    """测试各种追踪器的可用性"""
    print("测试追踪器可用性:")
    
    # 创建一个测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_bbox = (20, 20, 40, 40)
    
    # 测试新的API
    print("\n测试新API格式:")
    try:
        # 尝试新的创建方式
        tracker_types = ['KCF', 'CSRT', 'MIL']
        for tracker_type in tracker_types:
            try:
                print(f"\n测试 {tracker_type} 追踪器 (新API):")
                tracker = cv2.TrackerKCF.create() if tracker_type == 'KCF' else None
                if tracker_type == 'CSRT':
                    tracker = cv2.TrackerCSRT.create()
                elif tracker_type == 'MIL':
                    tracker = cv2.TrackerMIL.create()
                
                print(f"  - 创建成功: {tracker is not None}")
                
                if tracker is not None:
                    result = tracker.init(test_image, test_bbox)
                    print(f"  - init()返回值: {result} (类型: {type(result)})")
                    
                    if result:
                        success, bbox = tracker.update(test_image)
                        print(f"  - update()成功: {success}, bbox: {bbox}")
                    else:
                        print("  - init()失败")
            except Exception as e:
                print(f"  - {tracker_type} 追踪器 (新API) 不可用: {e}")
    except Exception as e:
        print(f"新API测试失败: {e}")
    
    # 测试旧的API
    print("\n测试旧API格式:")
    trackers_to_test = [
        ("KCF", lambda: cv2.TrackerKCF_create()),
        ("CSRT", lambda: cv2.TrackerCSRT_create()),
        ("MIL", lambda: cv2.TrackerMIL_create()),
    ]
    
    for name, creator in trackers_to_test:
        try:
            print(f"\n测试 {name} 追踪器:")
            tracker = creator()
            print(f"  - 创建成功: {tracker is not None}")
            
            if tracker is not None:
                # 测试初始化
                result = tracker.init(test_image, test_bbox)
                print(f"  - init()返回值: {result} (类型: {type(result)})")
                
                if result:
                    # 测试更新
                    success, bbox = tracker.update(test_image)
                    print(f"  - update()成功: {success}, bbox: {bbox}")
                else:
                    print("  - init()失败")
                    
        except Exception as e:
            print(f"  - {name} 追踪器不可用: {e}")

def test_simple_tracking():
    """简单的追踪测试"""
    print("\n" + "="*50)
    print("简单追踪测试:")
    
    # 创建两个稍微不同的测试图像
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (100, 100), (255, 255, 255), -1)
    
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img2, (55, 55), (105, 105), (255, 255, 255), -1)
    
    bbox = (50, 50, 50, 50)
    
    try:
        tracker = cv2.TrackerKCF_create()
        print(f"KCF追踪器创建: {tracker is not None}")
        
        if tracker is not None:
            # 测试在真实图像上的初始化
            result = tracker.init(img1, bbox)
            print(f"真实图像上init()结果: {result} (类型: {type(result)})")
            
            if result:
                success, new_bbox = tracker.update(img2)
                print(f"update()结果: {success}, 新边界框: {new_bbox}")
            else:
                print("初始化失败")
                
    except Exception as e:
        print(f"简单追踪测试失败: {e}")

if __name__ == "__main__":
    check_opencv_info()
    test_trackers()
    test_simple_tracking()
