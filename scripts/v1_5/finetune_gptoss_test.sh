#!/bin/bash

# GPT-OSS Vision Model Finetuning Script - 測試版
# 僅使用 COCO 資料集進行快速測試

# 設置環境變量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 模型參數
PRETRAIN_MODEL="./checkpoints/llava-gptoss-20b-pretrain"
VISION_TOWER="openai/clip-vit-large-patch14"

# 訓練參數 (測試用，較小的設定)
BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=2e-5
EPOCHS=1
MAX_STEPS=1000  # 限制步數，快速測試

# 路徑設置
DATA_PATH="./playground/data/llava_v1_5_mix665k_coco_only.json"
IMAGE_FOLDER="./playground/data"
OUTPUT_DIR="./checkpoints/llava-gptoss-20b-finetune-test"

echo "========================================="
echo "準備測試用資料集（僅 COCO）"
echo "========================================="

# 創建只包含 COCO 資料的 JSON 檔案
python3 << EOF
import json

print("載入完整資料集...")
with open('./playground/data/llava_v1_5_mix665k.json') as f:
    data = json.load(f)

print(f"原始總樣本數: {len(data)}")

# 只保留 COCO 資料
coco_data = [item for item in data if 'image' in item and item['image'].startswith('coco/')]
print(f"COCO 樣本數: {len(coco_data)}")

# 為了快速測試，可以只取前 10000 個樣本
test_data = coco_data[:10000]
print(f"測試用樣本數: {len(test_data)}")

# 保存測試用資料
output_path = './playground/data/llava_v1_5_mix665k_coco_only.json'
with open(output_path, 'w') as f:
    json.dump(test_data, f)

print(f"測試資料已保存到: {output_path}")
EOF

echo ""
echo "========================================="
echo "開始微調訓練（測試模式）"
echo "========================================="
echo "配置:"
echo "  - 資料集: COCO only (10,000 樣本)"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Gradient Accumulation: ${GRAD_ACCUM}"
echo "  - Max Steps: ${MAX_STEPS}"
echo "  - 輸出目錄: ${OUTPUT_DIR}"
echo "========================================="
echo ""

# 第二階段：全模型微調（測試版）
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${PRETRAIN_MODEL} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

echo ""
echo "========================================="
echo "測試訓練完成！"
echo "========================================="
echo "如果測試成功，可以運行完整版本:"
echo "  ./scripts/v1_5/finetune_gptoss.sh"
echo "========================================="
