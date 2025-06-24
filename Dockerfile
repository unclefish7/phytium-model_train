FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg git libgl1 unzip wget \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 安装 Python 依赖
COPY yolov5/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install timm && \
    python --version
