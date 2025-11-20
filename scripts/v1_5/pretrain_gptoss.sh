#!/bin/bash

# GPT-OSS Vision Model Training Script
# 使用 OpenAI GPT-OSS-20B 作為基礎模型

# 設置環境變量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 模型參數
MODEL_VERSION="gptoss-v1"
BASE_MODEL="openai/gpt-oss-20b"
VISION_TOWER="openai/clip-vit-large-patch14"

# 訓練參數
BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=2e-5
EPOCHS=1

# 路徑設置
DATA_PATH="./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
IMAGE_FOLDER="./playground/data/LLaVA-Pretrain"
OUTPUT_DIR="./checkpoints/llava-gptoss-20b-pretrain"

# 第一階段：預訓練 (只訓練 projector)
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${BASE_MODEL} \
    --version plain \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

echo "預訓練完成！模型保存在: ${OUTPUT_DIR}"
echo ""
echo "接下來運行微調腳本進行第二階段訓練"
