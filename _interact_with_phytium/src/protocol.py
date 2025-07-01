"""
通信协议封装模块
实现数据包的构建和解析，支持大数据包分片传输
"""
import struct
import time

# 协议常量
MAGIC = b'FSTM'
MAX_UDP_SIZE = 60000  # UDP最大数据包大小（保守设置）
FRAGMENT_HEADER_SIZE = 16  # 分片包头大小（与飞腾派接收端一致）

def build_packet(frame_id, jpeg_bytes):
    """
    构建发送数据包，支持大数据分片
    如果数据过大，返回分片包列表
    """
    data_size = len(jpeg_bytes)
    max_payload = MAX_UDP_SIZE - FRAGMENT_HEADER_SIZE
    
    # 如果数据小于限制，直接发送单个包
    if data_size <= max_payload:
        header = struct.pack('!4sIII', MAGIC, frame_id, data_size, 0)  # 16字节头部，0表示不分片
        return [header + jpeg_bytes]
    
    # 需要分片处理
    fragments = []
    total_fragments = (data_size + max_payload - 1) // max_payload  # 向上取整
    
    for i in range(total_fragments):
        start_pos = i * max_payload
        end_pos = min(start_pos + max_payload, data_size)
        fragment_data = jpeg_bytes[start_pos:end_pos]
        
        # 分片包头：magic(4) + frame_id(4) + total_size(4) + fragment_info(4) = 16字节
        # fragment_info: 高16位=总分片数，低16位=当前分片索引
        fragment_info = (total_fragments << 16) | i
        header = struct.pack('!4sIII', MAGIC, frame_id, data_size, fragment_info)  # 16字节头部
        fragments.append(header + fragment_data)
    
    return fragments

def parse_packet(data):
    """
    解析接收到的数据包
    返回：(frame_id, jpeg_bytes, is_complete) 或 None
    """
    if len(data) < FRAGMENT_HEADER_SIZE:
        return None
    
    try:
        magic, frame_id, total_size, fragment_info = struct.unpack('!4sIII', data[:FRAGMENT_HEADER_SIZE])
        if magic != MAGIC:
            return None
        
        fragment_data = data[FRAGMENT_HEADER_SIZE:]
        
        # 检查是否为分片包
        if fragment_info == 0:
            # 单个完整包
            return frame_id, fragment_data, True
        else:
            # 分片包，需要组装
            total_fragments = fragment_info >> 16
            current_fragment = fragment_info & 0xFFFF
            return frame_id, fragment_data, False, total_fragments, current_fragment
            
    except:
        return None


class FragmentAssembler:
    """
    分片数据包组装器
    """
    def __init__(self):
        self.fragments = {}  # frame_id -> {total_size, total_fragments, received_fragments}
    
    def add_fragment(self, frame_id, fragment_data, total_size, total_fragments, fragment_index):
        """
        添加分片数据
        返回：(is_complete, complete_data) 或 (False, None)
        """
        if frame_id not in self.fragments:
            self.fragments[frame_id] = {
                'total_size': total_size,
                'total_fragments': total_fragments,
                'received': {},
                'timestamp': time.time()
            }
        
        # 添加分片
        self.fragments[frame_id]['received'][fragment_index] = fragment_data
        
        # 检查是否接收完所有分片
        if len(self.fragments[frame_id]['received']) == total_fragments:
            # 组装完整数据
            complete_data = b''
            for i in range(total_fragments):
                if i in self.fragments[frame_id]['received']:
                    complete_data += self.fragments[frame_id]['received'][i]
                else:
                    return False, None  # 还有分片缺失
            
            # 清理已完成的帧数据
            del self.fragments[frame_id]
            return True, complete_data
        
        return False, None
    
    def cleanup_old_fragments(self, timeout=5.0):
        """
        清理超时的分片数据
        """
        import time
        current_time = time.time()
        expired_frames = []
        
        for frame_id, fragment_info in self.fragments.items():
            if current_time - fragment_info['timestamp'] > timeout:
                expired_frames.append(frame_id)
        
        for frame_id in expired_frames:
            del self.fragments[frame_id]
