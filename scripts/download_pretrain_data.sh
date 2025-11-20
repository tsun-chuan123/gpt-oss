#!/bin/bash

# 下載 LLaVA 預訓練資料集
# 使用 BLIP-Caption LAION/CC/SBU 558K 資料集

echo "開始下載 LLaVA 預訓練資料集..."

# 創建資料目錄
mkdir -p ./playground/data/LLaVA-Pretrain

# 下載資料集標註檔案
echo "下載資料集標註檔案 (blip_laion_cc_sbu_558k.json)..."
wget -P ./playground/data/LLaVA-Pretrain \
    https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json

# 下載圖片資料
echo "下載圖片資料 (images.zip)..."
echo "這個檔案較大，請耐心等待..."
wget -P ./playground/data/LLaVA-Pretrain \
    https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip

# 解壓縮圖片
echo "解壓縮圖片..."
cd ./playground/data/LLaVA-Pretrain
unzip -q images.zip
cd ../../..

echo "資料下載完成！"
echo "資料集位置: ./playground/data/LLaVA-Pretrain/"
echo "- 標註檔案: blip_laion_cc_sbu_558k.json"
echo "- 圖片資料夾: images/"
