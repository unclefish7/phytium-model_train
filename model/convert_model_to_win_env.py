import torch
import pathlib
import sys
import os

# 修复 PosixPath 不能在 Windows 上反序列化的问题
pathlib.PosixPath = pathlib.WindowsPath

sys.path.append(os.path.abspath('E:/tt100k/phytium-model_train/yolov5'))

# 原模型路径
input_path = 'model/latest_model_20250520/weights/best.pt'

# 加载模型
ckpt = torch.load(input_path, map_location='cpu', weights_only=False)

# 保存为新模型
output_path = 'model/latest_model_20250520/weights/best_win.pt'
torch.save(ckpt, output_path)

print(f"转换完成，保存为：{output_path}")
