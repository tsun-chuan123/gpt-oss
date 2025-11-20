# GPT-OSS Multimodal Training Docker Image
# 基於 NVIDIA CUDA 的官方映像,支援 GPU 加速訓練
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# 設定工作目錄
WORKDIR /workspace

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    # 基礎工具
    git \
    wget \
    curl \
    vim \
    build-essential \
    # Python 相關
    python3.10 \
    python3.10-dev \
    python3-pip \
    # 多媒體處理
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 網路工具
    netcat \
    # 清理快取
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 設定 Python 3.10 為預設版本
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 升級 pip
RUN pip install --upgrade pip setuptools wheel

# 複製 requirements.txt (利用 Docker 快取)
COPY requirements.txt /workspace/

# 安裝 Python 依賴
RUN pip install -r requirements.txt

# 安裝額外訓練工具
RUN pip install \
    # 分散式訓練
    deepspeed \
    # 實驗追蹤
    wandb \
    tensorboard \
    # Jupyter 支援
    jupyter \
    notebook \
    ipywidgets \
    # 程式碼品質
    black \
    flake8 \
    # 其他工具
    ninja \
    packaging

# 安裝 Flash Attention (可選,加速訓練)
# 如果編譯失敗或不需要加速,可以註解掉這行
RUN pip install flash-attn --no-build-isolation || echo "Flash Attention installation skipped"

# 複製專案程式碼
COPY . /workspace/

# 建立必要目錄
RUN mkdir -p /workspace/cache \
    /workspace/checkpoints \
    /workspace/logs \
    /workspace/data/images \
    /workspace/data/annotations

# 設定環境變數
ENV HF_HOME=/workspace/cache \
    TRANSFORMERS_CACHE=/workspace/cache \
    HF_DATASETS_CACHE=/workspace/cache \
    PYTHONPATH=/workspace:$PYTHONPATH \
    TORCH_HOME=/workspace/cache

# 暴露端口
EXPOSE 8888
EXPOSE 6006

# 設定工作目錄
WORKDIR /workspace

# 預設指令 (交互式 shell,方便訓練)
CMD ["/bin/bash"]
