"""
实时可视化模块
接收飞腾派回传结果并绘制检测框
支持多目标绘制和JSON数组格式
简化版本，不使用追踪器
"""
import socket
import json
import cv2
import numpy as np
from typing import List, Dict, Optional

# 配置参数
LISTEN_PORT = 9001

# 绘图配置
COLORS = [
    (0, 255, 0),    # 绿色
    (255, 0, 0),    # 蓝色
    (0, 0, 255),    # 红色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 品红色
    (0, 255, 255),  # 黄色
    (128, 0, 128),  # 紫色
    (255, 165, 0),  # 橙色
]
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_THICKNESS = 2


def draw_results(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    在图像上绘制检测结果
    
    Args:
        frame: 原始图像 (np.ndarray)
        detections: 检测结果列表，每个元素包含 x, y, w, h, label
        
    Returns:
        绘制了检测框的图像
    """
    if frame is None:
        return None
    
    # 复制图像避免修改原图
    vis_frame = frame.copy()
    
    # 绘制每个检测结果
    for i, detection in enumerate(detections):
        # 获取检测框参数
        x = int(detection.get('x', 0))
        y = int(detection.get('y', 0))
        w = int(detection.get('w', 0))
        h = int(detection.get('h', 0))
        label = detection.get('label', 'Unknown')
        
        # 选择颜色（循环使用预定义颜色）
        color = COLORS[i % len(COLORS)]
        
        # 绘制矩形框
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)
        
        # 计算标签背景框大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )
        
        # 绘制标签背景
        cv2.rectangle(
            vis_frame,
            (x, y - text_height - baseline - 5),
            (x + text_width, y),
            color,
            -1  # 填充
        )
        
        # 绘制标签文字
        cv2.putText(
            vis_frame, 
            label, 
            (x, y - baseline - 2), 
            FONT, 
            FONT_SCALE, 
            (255, 255, 255),  # 白色文字
            FONT_THICKNESS
        )
    
    # 在左上角显示检测数量
    if detections:
        info_text = f"Detections: {len(detections)}"
        cv2.putText(
            vis_frame, 
            info_text, 
            (10, 30), 
            FONT, 
            0.8, 
            (255, 255, 255), 
            2
        )
    
    return vis_frame


def receive_and_display(listen_port: int = LISTEN_PORT) -> None:
    """
    接收UDP数据并显示结果
    
    Args:
        listen_port: 监听端口号
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        sock.bind(('0.0.0.0', listen_port))
        print(f"开始监听端口: {listen_port}")
        print("按 ESC 或 'q' 退出")
        
        while True:
            try:
                # 接收数据
                data, addr = sock.recvfrom(8192)
                
                # 解析JSON数据
                detections = json.loads(data.decode('utf-8'))
                
                # 确保是列表格式
                if not isinstance(detections, list):
                    detections = [detections]
                
                print(f"收到 {len(detections)} 个检测结果")
                
                # 这里只是示例，实际使用时需要从主程序获取对应的frame
                # frame = get_original_frame(detections[0].get("frame_id"))
                # vis_frame = draw_results(frame, detections)
                # cv2.imshow("Detection Results", vis_frame)
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
            except Exception as e:
                print(f"接收数据错误: {e}")
            
            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC或q键退出
                break
                
    except Exception as e:
        print(f"监听端口失败: {e}")
    finally:
        sock.close()
        cv2.destroyAllWindows()


class ResultVisualizer:
    """
    结果可视化器类 - 保持向后兼容
    """
    def __init__(self, listen_port=LISTEN_PORT):
        self.listen_port = listen_port
        self.sock = None
        
    def __enter__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.listen_port))
        
        # 设置非阻塞模式
        self.sock.setblocking(False)
        
        print(f"可视化器已启动，监听端口: {self.listen_port}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sock:
            self.sock.close()
        cv2.destroyAllWindows()
    
    def receive_detections(self) -> Optional[List[Dict]]:
        """
        接收一次检测结果（非阻塞）
        飞腾派已经处理好数据清理，这里直接解析JSON
        
        Returns:
            检测结果列表或None
        """
        if not self.sock:
            return None
            
        try:
            data, addr = self.sock.recvfrom(8192)
            
            # 调试：打印接收到的数据
            print(f"收到响应数据: {len(data)} 字节")
            
            if len(data) == 0:
                print("收到空数据")
                return []
            
            # 直接解码UTF-8（飞腾派已经清理过数据）
            try:
                json_text = data.decode('utf-8')
                print(f"JSON文本: {json_text}")
                
                # 解析JSON
                detections = json.loads(json_text)
                
                # 确保是列表格式
                if not isinstance(detections, list):
                    detections = [detections]
                    
                print(f"JSON解析成功，检测到 {len(detections)} 个目标")
                return detections
                
            except UnicodeDecodeError as e:
                print(f"UTF-8解码失败: {e}")
                print(f"原始数据: {data[:50]}")
                return []
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print(f"问题文本: {json_text[:100] if 'json_text' in locals() else 'N/A'}")
                return []
            
        except socket.error as e:
            # 非阻塞模式下，没有数据时会抛出异常，这是正常的
            if e.errno == 10035:  # Windows下的WSAEWOULDBLOCK
                return None
            elif hasattr(e, 'errno') and e.errno in (11, 35):  # Linux下的EAGAIN或EWOULDBLOCK
                return None
            else:
                print(f"接收数据时发生网络错误: {e}")
                return None
        except Exception as e:
            print(f"接收检测结果失败: {e}")
            return None
    
    def display_frame_with_detections(self, frame: np.ndarray, detections: List[Dict], 
                                     window_name: str = "Detection Results") -> None:
        """
        显示带检测结果的图像
        
        Args:
            frame: 原始图像
            detections: 检测结果列表
            window_name: 窗口名称
        """
        vis_frame = draw_results(frame, detections)
        if vis_frame is not None:
            cv2.imshow(window_name, vis_frame)
            
    def show_both_windows(self, original_frame: np.ndarray, detections: List[Dict]) -> None:
        """
        同时显示原图和检测结果
        
        Args:
            original_frame: 原始图像
            detections: 检测结果列表
        """
        # 显示原图
        cv2.imshow("Original", original_frame)
        
        # 显示检测结果
        self.display_frame_with_detections(original_frame, detections, "Detections")
