"""
视频读取与JPEG编码模块
支持多种视频格式: MP4, AVI, MOV, MKV 等
支持关键帧发送和历史帧缓存
"""
import cv2
import time
import os
from collections import deque

# 配置参数
VIDEO_PATH = r"E:\tt100k\sample_video\output_40_58.avi"  # 视频文件路径，设为0则使用摄像头
FPS = 1  # 播放帧率
KEY_FRAME_INTERVAL = 2  # 关键帧间隔（每10帧发送一次关键帧）
FRAME_BUFFER_SIZE = 50  # 历史帧缓存大小

# 支持的视频格式
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

class VideoCapture:
    def __init__(self, source=VIDEO_PATH, fps=FPS):
        self.source = source
        self.fps = fps
        self.cap = None
        self.frame_id = 0
        self.total_frames = 0
        self.video_fps = 0
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)  # 历史帧缓存
        
    def __enter__(self):
        # 检查视频文件格式和存在性
        if isinstance(self.source, str) and self.source != "0":
            if not os.path.exists(self.source):
                raise RuntimeError(f"视频文件不存在: {self.source}")
            
            # 检查文件格式
            file_ext = os.path.splitext(self.source)[1].lower()
            if file_ext not in SUPPORTED_FORMATS:
                print(f"警告: 文件格式 {file_ext} 可能不被支持，支持的格式: {SUPPORTED_FORMATS}")
        
        # 创建VideoCapture对象
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")
        
        # 获取视频信息
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("视频信息:")
        print(f"  文件: {self.source}")
        print(f"  分辨率: {width}x{height}")
        print(f"  总帧数: {self.total_frames}")
        print(f"  原始帧率: {self.video_fps:.2f} FPS")
        print(f"  播放帧率: {self.fps} FPS")
        
        # 对于AVI格式，设置一些特殊参数以提高兼容性
        if isinstance(self.source, str) and self.source.lower().endswith('.avi'):
            # 尝试设置缓冲区大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.cap or not self.cap.isOpened():
            raise StopIteration
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            # 检查是否到达视频末尾
            if self.total_frames > 0 and self.frame_id >= self.total_frames:
                print(f"视频播放完成，共处理 {self.frame_id} 帧")
            else:
                print(f"读取帧失败，当前帧: {self.frame_id}")
            raise StopIteration
        
        # 对于某些AVI文件，可能需要检查帧的有效性
        if frame.size == 0:
            print(f"跳过空帧: {self.frame_id}")
            self.frame_id += 1
            return self.__next__()  # 递归调用获取下一帧
        
        # 缩放图像以减小数据量（保持纵横比）
        # 设置最小分辨率为1280x720，只对更大的图像进行缩放
        original_height, original_width = frame.shape[:2]
        min_width, min_height = 1280, 720  # 最小分辨率限制
        
        # 只有当图像比最小分辨率更大时才进行缩放
        if original_width > min_width or original_height > min_height:
            # 计算缩放比例，确保不小于最小分辨率
            scale_w = min_width / original_width if original_width > min_width else 1.0
            scale_h = min_height / original_height if original_height > min_height else 1.0
            scale = max(scale_w, scale_h)  # 选择较大的缩放比例以确保不小于最小尺寸
            
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 确保不小于最小分辨率
            new_width = max(new_width, min_width)
            new_height = max(new_height, min_height)
            
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            if self.frame_id == 0:  # 只在第一帧打印缩放信息
                print(f"图像缩放: {original_width}x{original_height} -> {new_width}x{new_height}")
        else:
            if self.frame_id == 0:
                print(f"图像尺寸: {original_width}x{original_height} (无需缩放)")
        
        # 编码为JPEG，使用中等质量以平衡文件大小和图像质量
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # 提高质量到75
        ret_encode, jpeg_bytes = cv2.imencode('.jpg', frame, encode_param)
        
        if not ret_encode:
            print(f"JPEG编码失败，帧: {self.frame_id}")
            self.frame_id += 1
            return self.__next__()  # 递归调用获取下一帧
        
        jpeg_bytes = jpeg_bytes.tobytes()
        
        # 将帧添加到历史缓存
        self.frame_buffer.append((self.frame_id, frame.copy()))
        
        # 显示数据大小信息
        if self.frame_id == 0:
            print(f"JPEG数据大小: {len(jpeg_bytes)} 字节")
        
        # 显示进度
        if self.frame_id % 10 == 0:
            progress = (self.frame_id / self.total_frames * 100) if self.total_frames > 0 else 0
            print(f"处理进度: {self.frame_id}/{self.total_frames} ({progress:.1f}%) - 数据大小: {len(jpeg_bytes)} 字节")
        
        # 返回格式: (frame_id, jpeg_bytes, raw_frame)
        # 现在总是返回jpeg_bytes，由main函数决定是否发送
        current_frame_id = self.frame_id
        self.frame_id += 1
        
        # 控制帧率 - 使用更短的延迟避免程序卡住
        if self.fps > 0:
            delay = 1.0 / self.fps
            if delay > 0.1:  # 如果延迟超过100ms，则分段延迟
                time.sleep(0.1)
            else:
                time.sleep(delay)
        
        return (current_frame_id, jpeg_bytes, frame.copy())
    
    def get_video_info(self):
        """
        获取视频信息
        """
        if not self.cap:
            return None
        
        return {
            'total_frames': self.total_frames,
            'fps': self.video_fps,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'current_frame': self.frame_id
        }
    
    def get_frame_from_buffer(self, frame_id):
        """
        从缓存中获取指定ID的帧
        
        Args:
            frame_id: 帧ID
            
        Returns:
            对应的帧图像或None
        """
        for fid, frame in self.frame_buffer:
            if fid == frame_id:
                return frame.copy()
        return None
    
    def get_frame_buffer(self):
        """
        获取当前帧缓存
        """
        return list(self.frame_buffer)
