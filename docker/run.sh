#!/bin/bash
# GPT-OSS Vision Model Docker 啟動並進入容器

set -e

# 切換到 docker 目錄
cd "$(dirname "$0")"

# 檢查是否需要建置
if ! docker images | grep -q "gpt-oss"; then
    echo "📦 映像不存在,開始建置..."
    bash build.sh
fi

# 檢查容器是否正在運行
if docker compose ps gpt-oss-training | grep -q "Up"; then
    echo "✅ 容器已在運行,直接進入..."
else
    echo "🚀 啟動容器..."
    docker compose up -d gpt-oss-training
    echo "⏳ 等待容器啟動..."
    sleep 3
fi

# 直接進入容器
echo "🔗 進入 GPT-OSS 訓練容器..."
echo ""
echo "💡 在容器內可以執行:"
echo "   Stage 1: bash scripts/run_align.sh"
echo "   Stage 2: bash scripts/run_sft.sh"
echo ""
docker compose exec gpt-oss-training bash
