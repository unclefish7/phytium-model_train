"""
UDP发送模块
将视频帧发送到飞腾派主核，支持大数据包分片传输
"""
import socket
import time
from protocol import build_packet

# 配置参数
FT_BOARD_IP = "10.243.4.43"
FT_BOARD_PORT = 9000

class UDPSender:
    def __init__(self, target_ip=FT_BOARD_IP, target_port=FT_BOARD_PORT):
        self.target_ip = target_ip
        self.target_port = target_port
        self.sock = None
        
    def __enter__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 设置更大的发送缓冲区
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB
            buffer_size = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            print(f"UDP发送缓冲区大小: {buffer_size} 字节")
        except Exception as e:
            print(f"设置发送缓冲区失败: {e}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sock:
            self.sock.close()
    
    def send_frame(self, frame_id, jpeg_bytes):
        """
        发送单帧数据，支持自动分片
        """
        if not self.sock:
            raise RuntimeError("UDP socket未初始化")
        
        # 获取数据包列表（可能是分片的）
        packets = build_packet(frame_id, jpeg_bytes)
        
        success_count = 0
        total_packets = len(packets)
        
        print(f"帧 {frame_id}: {len(jpeg_bytes)} 字节, 分为 {total_packets} 个包")
        
        for i, packet in enumerate(packets):
            try:
                self.sock.sendto(packet, (self.target_ip, self.target_port))
                success_count += 1
                
                # 分片包之间稍微延迟，避免网络拥塞
                if total_packets > 1 and i < total_packets - 1:
                    time.sleep(0.001)  # 1ms延迟
                    
            except Exception as e:
                print(f"发送帧 {frame_id} 分片 {i+1}/{total_packets} 失败: {e}")
                return False
        
        if success_count == total_packets:
            print(f"已发送帧 {frame_id}, 总大小: {len(jpeg_bytes)} 字节 ({total_packets} 个包)")
            return True
        else:
            print(f"帧 {frame_id} 发送不完整: {success_count}/{total_packets}")
            return False
