"""
主程序入口
一键启动视频采集、发送和可视化
简单展示模式，接收飞腾派返回的检测结果并显示
"""
import cv2
import time
import json
import os
from capture import VideoCapture
from sender import UDPSender
from visualizer import ResultVisualizer, draw_results

def get_output_video_path(input_path):
    """
    根据输入视频路径生成输出视频路径
    例如: input.avi -> input_detected.avi
    """
    if isinstance(input_path, str) and input_path != "0":
        # 分离路径、文件名和扩展名
        dir_path = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        
        # 生成新的文件名
        output_filename = f"{name}_detected{ext}"
        output_path = os.path.join(dir_path, output_filename)
        
        return output_path
    else:
        # 如果是摄像头输入，使用默认文件名
        return "detected_output.avi"

def create_video_writer(output_path, fps, frame_width, frame_height, codec='XVID'):
    """
    创建视频写入器，支持多种编码格式
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # 验证写入器是否创建成功
        if not writer.isOpened():
            print(f"警告: 使用 {codec} 编码器创建视频写入器失败，尝试备用编码器...")
            # 尝试备用编码器
            backup_codecs = ['MJPG', 'mp4v', 'XVID']
            for backup_codec in backup_codecs:
                if backup_codec != codec:
                    try:
                        fourcc_backup = cv2.VideoWriter_fourcc(*backup_codec)
                        writer = cv2.VideoWriter(output_path, fourcc_backup, fps, (frame_width, frame_height))
                        if writer.isOpened():
                            print(f"成功使用备用编码器: {backup_codec}")
                            return writer
                    except:
                        continue
            
            print("所有编码器都无法创建视频写入器")
            return None
        
        return writer
        
    except Exception as e:
        print(f"创建视频写入器时出错: {e}")
        return None

def main():
    print("=== 飞腾派通信程序启动 ===")
    
    # ===== 配置参数 =====
    # 视频输入配置
    VIDEO_PATH = r"E:\tt100k\sample_video\output_0309_0335.avi"  # 视频文件路径，设为0则使用摄像头
    VIDEO_PLAYBACK_FPS = 5  # 视频播放帧率（用于控制播放速度）
    
    # UDP发送配置
    SEND_EVERY_N_FRAMES = 5  # 每隔N帧发送一次到飞腾派（关键帧间隔）
    UDP_SEND_DELAY = 1.5  # UDP发送间隔（秒），控制发送频率
    
    # 视频保存配置
    SAVE_VIDEO = True  # 是否保存检测结果视频
    OUTPUT_VIDEO_CODEC = 'XVID'  # 视频编码器: 'XVID', 'MJPG', 'mp4v'
    OUTPUT_VIDEO_FPS = 10  # 输出视频的帧率（独立于输入视频）
    
    # 处理控制配置
    MAX_FRAME_CACHE = 200  # 最大帧缓存数量
    MAX_WAIT_CYCLES = 500  # 等待检测结果的最大循环次数
    
    # 使用局部变量来控制视频保存
    save_video_enabled = SAVE_VIDEO
    
    print("配置参数:")
    print(f"  视频路径: {VIDEO_PATH}")
    print(f"  播放帧率: {VIDEO_PLAYBACK_FPS} FPS")
    print(f"  发送间隔: 每 {SEND_EVERY_N_FRAMES} 帧")
    print(f"  UDP发送延迟: {UDP_SEND_DELAY} 秒")
    print(f"  输出视频帧率: {OUTPUT_VIDEO_FPS} FPS")
    print(f"  保存视频: {'是' if save_video_enabled else '否'}")
    
    try:
        # 初始化结果可视化器
        with ResultVisualizer() as visualizer:
            # 初始化发送器
            with UDPSender() as sender:
                # 初始化视频采集
                with VideoCapture(VIDEO_PATH, VIDEO_PLAYBACK_FPS) as capture:
                    
                    print(f"开始处理视频: {VIDEO_PATH}")
                    print(f"播放帧率: {VIDEO_PLAYBACK_FPS} FPS")
                    print(f"发送间隔: 每 {SEND_EVERY_N_FRAMES} 帧")
                    print("按 'q' 或 ESC 退出程序")
                    
                    frame_count = 0
                    frame_cache = {}  # {frame_id: raw_frame}
                    processed_frames = {}  # {frame_id: processed_frame} 用于保存视频
                    
                    # 初始化视频写入器
                    video_writer = None
                    output_video_path = None
                    if save_video_enabled:
                        output_video_path = get_output_video_path(VIDEO_PATH)
                        print(f"检测结果将保存到: {output_video_path}")
                    
                    print("开始视频处理循环...")
                    
                    # 主循环：发送视频帧并处理检测结果
                    video_finished = False
                    
                    while not video_finished:
                        # 发送视频帧部分
                        try:
                            frame_id, jpeg_bytes, raw_frame = next(capture)
                            frame_count += 1
                            
                            # 判断是否为关键帧（基于帧间隔设置）
                            is_key_frame = (frame_id % SEND_EVERY_N_FRAMES == 0)
                            
                            if is_key_frame and jpeg_bytes is not None:
                                # 缓存关键帧，以便后续显示检测结果
                                frame_cache[frame_id] = raw_frame.copy()
                                
                                # 发送关键帧到飞腾派
                                success = sender.send_frame(frame_id, jpeg_bytes)
                                if success:
                                    print(f"发送关键帧 {frame_id}")
                                    # 控制发送频率
                                    if UDP_SEND_DELAY > 0:
                                        time.sleep(UDP_SEND_DELAY)
                                else:
                                    print(f"发送关键帧 {frame_id} 失败")
                                    
                                # 限制缓存大小，避免内存溢出
                                if len(frame_cache) > MAX_FRAME_CACHE:
                                    oldest_frame_id = min(frame_cache.keys())
                                    del frame_cache[oldest_frame_id]
                                    print(f"清理过期缓存帧: {oldest_frame_id}")
                                    
                        except StopIteration:
                            print("视频读取完成")
                            video_finished = True
                        
                        # 处理检测结果部分（非阻塞）
                        detections = visualizer.receive_detections()
                        if detections and len(detections) > 0:
                            detection_frame_id = detections[0].get('frame_id', frame_count)
                            print(f"收到帧 {detection_frame_id} 的检测结果: {len(detections)} 个目标")
                            
                            # 从缓存中获取对应的帧来显示检测结果
                            if detection_frame_id in frame_cache:
                                cached_frame = frame_cache[detection_frame_id]
                                
                                # 绘制检测结果
                                processed_frame = draw_results(cached_frame, detections)
                                
                                # 显示结果
                                visualizer.display_frame_with_detections(cached_frame, detections)
                                print(f"显示帧 {detection_frame_id} 的检测结果")
                                
                                # 保存处理后的帧用于视频输出
                                if save_video_enabled and processed_frame is not None:
                                    processed_frames[detection_frame_id] = processed_frame.copy()
                                    
                                    # 初始化视频写入器（使用第一帧的尺寸）
                                    if video_writer is None:
                                        frame_height, frame_width = processed_frame.shape[:2]
                                        video_writer = create_video_writer(
                                            output_video_path,
                                            OUTPUT_VIDEO_FPS,
                                            frame_width,
                                            frame_height,
                                            OUTPUT_VIDEO_CODEC
                                        )
                                        if video_writer is not None:
                                            print(f"成功初始化视频写入器: {frame_width}x{frame_height}, {OUTPUT_VIDEO_FPS} FPS")
                                        else:
                                            print("视频写入器初始化失败，将不保存视频")
                                            save_video_enabled = False
                                
                                # 清理已使用的缓存帧
                                del frame_cache[detection_frame_id]
                            else:
                                print(f"警告: 未找到帧 {detection_frame_id} 在缓存中")
                        
                        # 检查用户是否要退出
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # q键或ESC键
                            print("用户请求退出")
                            break
                    
                    print("视频发送完成，等待剩余检测结果...")
                    
                    # 继续接收检测结果，直到处理完所有缓存的帧
                    max_wait_iterations = MAX_WAIT_CYCLES  # 使用配置的等待循环次数
                    wait_iterations = 0
                    
                    while len(frame_cache) > 0 and wait_iterations < max_wait_iterations:
                        detections = visualizer.receive_detections()
                        if detections and len(detections) > 0:
                            detection_frame_id = detections[0].get('frame_id', -1)
                            print(f"收到剩余检测结果: 帧 {detection_frame_id}, {len(detections)} 个目标")
                            
                            # 从缓存中获取对应的帧来显示检测结果
                            if detection_frame_id in frame_cache:
                                cached_frame = frame_cache[detection_frame_id]
                                
                                # 绘制检测结果
                                processed_frame = draw_results(cached_frame, detections)
                                
                                # 显示结果
                                visualizer.display_frame_with_detections(cached_frame, detections)
                                print(f"显示剩余帧 {detection_frame_id} 的检测结果")
                                
                                # 保存处理后的帧用于视频输出
                                if save_video_enabled and processed_frame is not None:
                                    processed_frames[detection_frame_id] = processed_frame.copy()
                                    
                                    # 初始化视频写入器（如果还没有初始化）
                                    if video_writer is None:
                                        frame_height, frame_width = processed_frame.shape[:2]
                                        video_writer = create_video_writer(
                                            output_video_path,
                                            OUTPUT_VIDEO_FPS,
                                            frame_width,
                                            frame_height,
                                            OUTPUT_VIDEO_CODEC
                                        )
                                        if video_writer is not None:
                                            print(f"成功初始化视频写入器: {frame_width}x{frame_height}, {OUTPUT_VIDEO_FPS} FPS")
                                        else:
                                            print("视频写入器初始化失败，将不保存视频")
                                            save_video_enabled = False
                                
                                # 清理已使用的缓存帧
                                del frame_cache[detection_frame_id]
                            else:
                                print(f"警告: 剩余检测结果帧 {detection_frame_id} 不在缓存中")
                        
                        # 检查用户是否要退出
                        key = cv2.waitKey(100) & 0xFF  # 等待100ms
                        if key == ord('q') or key == 27:
                            break
                            
                        wait_iterations += 1
                    
                    # 清理剩余的缓存
                    remaining_frames = len(frame_cache)
                    if remaining_frames > 0:
                        print(f"清理 {remaining_frames} 个未处理的缓存帧")
                        # 显示所有剩余帧的帧ID
                        remaining_ids = sorted(frame_cache.keys())
                        print(f"未处理的帧ID: {remaining_ids}")
                    
                    # 保存视频
                    if save_video_enabled and video_writer is not None and len(processed_frames) > 0:
                        print(f"开始保存视频，共 {len(processed_frames)} 帧...")
                        
                        # 按帧ID排序并写入视频
                        sorted_frame_ids = sorted(processed_frames.keys())
                        for frame_id in sorted_frame_ids:
                            processed_frame = processed_frames[frame_id]
                            video_writer.write(processed_frame)
                        
                        # 释放视频写入器
                        video_writer.release()
                        print(f"视频保存完成: {output_video_path}")
                        print(f"保存了 {len(processed_frames)} 帧检测结果")
                        
                        # 验证保存的视频文件
                        if os.path.exists(output_video_path):
                            file_size = os.path.getsize(output_video_path)
                            print(f"输出视频文件大小: {file_size / (1024*1024):.2f} MB")
                        else:
                            print("警告: 输出视频文件创建失败")
                    elif save_video_enabled and len(processed_frames) == 0:
                        print("没有检测结果帧需要保存")
                    
                    print("所有处理完成")
                    
                    # 等待用户关闭窗口
                    print("按任意键关闭窗口...")
                    cv2.waitKey(0)
                    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保视频写入器被正确释放
        try:
            if 'video_writer' in locals() and video_writer is not None:
                video_writer.release()
                print("视频写入器已释放")
        except:
            pass
        
        cv2.destroyAllWindows()
        print("程序结束")


def demo_draw_results():
    """
    演示draw_results函数的使用
    """
    # 创建示例图像
    frame = cv2.imread("test_image.jpg")  # 需要替换为实际图像路径
    if frame is None:
        # 创建一个示例图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (590, 430), (64, 64, 64), -1)
    
    # 示例检测结果
    detections = [
        {"frame_id": 1, "x": 100, "y": 120, "w": 80, "h": 60, "label": "STOP"},
        {"frame_id": 1, "x": 300, "y": 150, "w": 90, "h": 70, "label": "SPEED_LIMIT"},
        {"frame_id": 1, "x": 200, "y": 300, "w": 60, "h": 40, "label": "WARNING"}
    ]
    
    # 绘制检测结果
    vis_frame = draw_results(frame, detections)
    
    # 显示结果
    cv2.imshow("Demo - Detection Results", vis_frame)
    print("按任意键退出演示...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    import numpy as np
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_draw_results()
    else:
        main()
