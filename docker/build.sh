#!/bin/bash
# GPT-OSS Vision Model Docker 建置腳本

set -e  # 遇到錯誤立即停止

echo "🔨 開始建置 GPT-OSS Vision Model Docker 映像..."

# 切換到專案根目錄
cd "$(dirname "$0")/.."

# 建置 Docker 映像
docker compose -f docker/docker-compose.yml build --no-cache

echo "✅ 建置完成!"
echo ""
echo "💡 使用方式:"
echo "   啟動並進入容器: bash run.sh"
echo "   或直接: cd docker && docker compose up -d"
echo "   停止服務: bash stop.sh"