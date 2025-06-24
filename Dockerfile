FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg git htop libgl1 net-tools tmux unzip wget \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 安装 Python 依赖
COPY yolov5/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install timm && \
    python --version

# 写入 tmux 配置文件，开启鼠标支持 + 分屏热键优化
RUN echo '# 开启鼠标支持\n\
set -g mouse on\n\
\n\
# 使用 Vi 模式\n\
setw -g mode-keys vi\n\
\n\
# 设置状态栏更新频率和颜色\n\
set -g status-interval 5\n\
set -g status-bg black\n\
set -g status-fg white\n\
\n\
# 更舒服的分屏快捷键\n\
bind | split-window -h\n\
bind - split-window -v\n\
unbind "\""\n\
unbind "%"\n\
\n\
# 启用256色\n\
set -g default-terminal "screen-256color"' > /root/.tmux.conf
