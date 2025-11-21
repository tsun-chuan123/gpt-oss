#!/bin/bash

# LLaVA 1.5 微調資料下載腳本
# 這個腳本會下載所有需要的圖片資料集

set -e

DATA_DIR="/workspace/playground/data"
cd ${DATA_DIR}

echo "========================================="
echo "LLaVA 1.5 微調資料下載"
echo "========================================="

# 1. 檢查 JSON 註解檔案
echo ""
echo "[1/6] 檢查訓練註解檔案..."
if [ -f "llava_v1_5_mix665k.json" ]; then
    echo "✓ llava_v1_5_mix665k.json 已存在 ($(du -h llava_v1_5_mix665k.json | cut -f1))"
else
    echo "下載 llava_v1_5_mix665k.json..."
    wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json
    echo "✓ 下載完成"
fi

# 2. 下載 COCO train2017
# echo ""
# echo "[2/6] 下載 COCO train2017..."
# if [ -d "coco/train2017" ]; then
#     echo "✓ coco/train2017 已存在 ($(find coco/train2017 -type f | wc -l) 個檔案)"
# else
#     echo "下載 COCO train2017 (約 18GB)..."
#     mkdir -p coco
#     wget http://images.cocodataset.org/zips/train2017.zip
#     echo "解壓縮中..."
#     unzip -q train2017.zip -d coco/
#     rm train2017.zip
#     echo "✓ COCO train2017 下載完成"
# fi

# # 3. 下載 GQA images
# echo ""
# echo "[3/6] 下載 GQA images..."
# if [ -d "gqa/images" ]; then
#     echo "✓ gqa/images 已存在 ($(find gqa/images -type f | wc -l) 個檔案)"
# else
#     echo "下載 GQA images (約 20GB)..."
#     mkdir -p gqa
#     wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O gqa_images.zip
#     echo "解壓縮中..."
#     unzip -q gqa_images.zip -d gqa/
#     rm gqa_images.zip
#     echo "✓ GQA images 下載完成"
# fi

# # 4. 下載 OCR-VQA images
# echo ""
# echo "[4/6] 下載 OCR-VQA images..."
# if [ -d "ocr_vqa/images" ]; then
#     echo "✓ ocr_vqa/images 已存在 ($(find ocr_vqa/images -type f | wc -l) 個檔案)"
# else
#     echo "⚠️  OCR-VQA 需要手動從 Google Drive 下載"
#     echo "請訪問: https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_"
#     echo "下載後解壓到 ${DATA_DIR}/ocr_vqa/images/"
#     echo "注意：所有檔案需要轉換為 .jpg 格式"
# fi

# # 5. 下載 TextVQA train_val_images
# echo ""
# echo "[5/6] 下載 TextVQA train_val_images..."
# if [ -d "textvqa/train_images" ]; then
#     echo "✓ textvqa/train_images 已存在 ($(find textvqa/train_images -type f | wc -l) 個檔案)"
# else
#     echo "下載 TextVQA train_val_images (約 6GB)..."
#     mkdir -p textvqa
#     wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
#     echo "解壓縮中..."
#     unzip -q train_val_images.zip -d textvqa/
#     rm train_val_images.zip
#     echo "✓ TextVQA train_val_images 下載完成"
# fi

# # 6. 下載 Visual Genome (VG)
# echo ""
# echo "[6/6] 下載 Visual Genome images..."
# VG_COMPLETE=true
# if [ ! -d "vg/VG_100K" ]; then
#     VG_COMPLETE=false
#     echo "下載 Visual Genome Part 1 (約 15GB)..."
#     mkdir -p vg
#     wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O vg_part1.zip
#     echo "解壓縮中..."
#     unzip -q vg_part1.zip -d vg/
#     rm vg_part1.zip
# fi

# if [ ! -d "vg/VG_100K_2" ]; then
#     VG_COMPLETE=false
#     echo "下載 Visual Genome Part 2 (約 5GB)..."
#     mkdir -p vg
#     wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O vg_part2.zip
#     echo "解壓縮中..."
#     unzip -q vg_part2.zip -d vg/
#     rm vg_part2.zip
# fi

# if [ "$VG_COMPLETE" = true ]; then
#     echo "✓ vg/VG_100K 已存在 ($(find vg/VG_100K -type f | wc -l) 個檔案)"
#     echo "✓ vg/VG_100K_2 已存在 ($(find vg/VG_100K_2 -type f | wc -l) 個檔案)"
# else
#     echo "✓ Visual Genome 下載完成"
# fi

echo ""
echo "========================================="
echo "資料下載總結"
echo "========================================="

# 檢查目錄結構
echo ""
echo "目錄結構檢查:"
echo "$(tree -L 2 -d ${DATA_DIR} 2>/dev/null || find ${DATA_DIR} -maxdepth 2 -type d)"

echo ""
echo "資料集統計:"
[ -f "llava_v1_5_mix665k.json" ] && echo "  ✓ 訓練註解: llava_v1_5_mix665k.json (665,298 樣本)"
[ -d "coco/train2017" ] && echo "  ✓ COCO: $(find coco/train2017 -type f | wc -l) 張圖片"
[ -d "gqa/images" ] && echo "  ✓ GQA: $(find gqa/images -type f | wc -l) 張圖片"
[ -d "ocr_vqa/images" ] && echo "  ✓ OCR-VQA: $(find ocr_vqa/images -type f 2>/dev/null | wc -l) 張圖片"
[ -d "textvqa/train_images" ] && echo "  ✓ TextVQA: $(find textvqa/train_images -type f | wc -l) 張圖片"
[ -d "vg/VG_100K" ] && echo "  ✓ VG Part1: $(find vg/VG_100K -type f | wc -l) 張圖片"
[ -d "vg/VG_100K_2" ] && echo "  ✓ VG Part2: $(find vg/VG_100K_2 -type f | wc -l) 張圖片"

echo ""
echo "磁碟使用量:"
du -sh ${DATA_DIR}

echo ""
echo "========================================="
echo "準備完成！"
echo "========================================="
echo ""
echo "⚠️  注意事項:"
echo "1. 如果 OCR-VQA 未下載，請手動從 Google Drive 下載"
echo "2. 確保所有圖片路徑與 llava_v1_5_mix665k.json 中的路徑一致"
echo "3. 預計總共需要 ~70-80GB 的磁碟空間"
echo ""
echo "現在可以運行微調腳本:"
echo "  chmod +x scripts/v1_5/finetune_gptoss.sh"
echo "  ./scripts/v1_5/finetune_gptoss.sh"
