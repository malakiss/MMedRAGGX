#!/usr/bin/env bash
# Generate radiology reports with fine-tuned MedGemma

set -euo pipefail

CHECKPOINT="./checkpoints/medgemma_dpo"
RETRIEVED="./output_rag_radgraph.json"
IMG_ROOT="/path/to/iu_xray/images"
OUTPUT="./results_medgemma.json"

cd "$(dirname "$0")/.." || exit 1

CUDA_VISIBLE_DEVICES=0 python ./train/medgemma/inference_medgemma.py \
    --checkpoint   "$CHECKPOINT" \
    --retrieved    "$RETRIEVED" \
    --img_root     "$IMG_ROOT" \
    --output       "$OUTPUT" \
    --max_new_tokens 256 \
    --temperature  0.1 \
    --use_4bit
