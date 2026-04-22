#!/usr/bin/env bash
# MedGemma DPO fine-tuning — radiology reports, MIMIC + IU X-Ray
#
# Single GPU:     bash scripts/train_dpo_medgemma.sh
# 2x T4 (Kaggle): CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_dpo_medgemma.sh

set -euo pipefail

# ── Edit these paths ─────────────────────────────────────────────────────────
DATA_PATH="./data/training/alignment/radiology/radiology_report_radgraph.json"
IMAGE_FOLDER="/kaggle/input/iu-xray/images"   # update to your Kaggle dataset path
OUTPUT_DIR="./checkpoints/medgemma_dpo"

# Remap image_root baked into the JSON → your actual paths
IMAGE_ROOT_REMAP="\
/home/wenhao/Datasets/med/rad/iu_xray/images:/kaggle/input/iu-xray/images,\
/home/wenhao/Datasets/med/rad/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files:/kaggle/input/mimic-cxr/files"
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$OUTPUT_DIR"

# Count GPUs from CUDA_VISIBLE_DEVICES (default: GPU 0 only)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "Launching on $NUM_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"

COMMON_ARGS=(
    ./train/medgemma/train_dpo_medgemma.py
    --model_name_or_path    google/medgemma-4b-it
    --data_path             "$DATA_PATH"
    --image_folder          "$IMAGE_FOLDER"
    --image_root_remap      "$IMAGE_ROOT_REMAP"
    --output_dir            "$OUTPUT_DIR"
    --max_length            1024
    --use_4bit              True
    --lora_enable           True
    --lora_r                128
    --lora_alpha            256
    --lora_dropout          0.05
    --beta                  0.1
    --num_train_epochs      3
    --per_device_train_batch_size  1
    --gradient_accumulation_steps  4
    --learning_rate         1e-7
    --lr_scheduler_type     cosine
    --warmup_ratio          0.03
    --weight_decay          0.0
    --bf16                  True
    --gradient_checkpointing True
    --logging_steps         10
    --save_steps            100
    --save_total_limit      2
    --report_to             wandb
    --run_name              medgemma_dpo_radiology
    --dataloader_num_workers 2
    --remove_unused_columns False
)

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU: accelerate DDP — compatible with QLoRA (DeepSpeed is not)
    accelerate launch \
        --num_processes "$NUM_GPUS" \
        --mixed_precision bf16 \
        "${COMMON_ARGS[@]}"
else
    python "${COMMON_ARGS[@]}"
fi
