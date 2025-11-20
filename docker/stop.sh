#!/bin/bash
# GPT-OSS Vision Model Docker 停止腳本

set -e

echo "🛑 停止 GPT-OSS 服務..."

# 切換到 docker 目錄
cd "$(dirname "$0")"

# 停止所有服務
docker compose down

echo "✅ 所有服務已停止"
echo ""
echo "💡 提示:"
echo "   重新啟動: bash run.sh"
echo "   完全清理 (包含 volumes): docker compose down -v"
