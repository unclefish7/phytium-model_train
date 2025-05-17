import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {'是' if torch.cuda.is_available() else '否'}")
if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")