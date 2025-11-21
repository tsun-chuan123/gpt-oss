#!/bin/bash

# GPT-OSS Vision Model Finetuning Script
# 第二階段：全模型微調

# 設置環境變量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 模型參數
BASE_MODEL="openai/gpt-oss-20b"
PRETRAIN_MM_ADAPTER="./checkpoints/llava-gptoss-20b-pretrain/mm_projector.bin"
VISION_TOWER="openai/clip-vit-large-patch14"

# 訓練參數
BATCH_SIZE=1          # 減小到 1
GRAD_ACCUM=16         # 增加梯度累積以保持有效 batch size
LEARNING_RATE=2e-5
EPOCHS=1

# 路徑設置
DATA_PATH="./playground/data/llava_v1_5_mix665k.json"
IMAGE_FOLDER="./playground/data"
OUTPUT_DIR="./checkpoints/llava-gptoss-20b-finetune"

# 第二階段：全模型微調
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path ${BASE_MODEL} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --pretrain_mm_mlp_adapter ${PRETRAIN_MM_ADAPTER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --save_strategy "steps" \
    --save_steps 50000 \
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

echo "微調完成！最終模型保存在: ${OUTPUT_DIR}"
