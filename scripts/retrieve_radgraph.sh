#!/usr/bin/env bash
# OpenCLIP top-3 retrieval + RadGraph-XL entity extraction
# Outputs a JSONL with radgraph_context field ready for MedGemma

set -euo pipefail

# ── Edit these paths ─────────────────────────────────────────────────────────
IMG_ROOT="/path/to/iu_xray/images"
TRAIN_JSON="./data/training/retriever/radiology/rad_iu.json"
EVAL_JSON="./data/test/report/mimic_test.json"       # change to iuxray for val
CLIP_CHECKPOINT="./train/open_clip/src/logs/epoch_360.pt"
OUTPUT_PATH="./output_rag_radgraph.json"
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.." || exit 1

CUDA_VISIBLE_DEVICES=0 python ./train/open_clip/src/retrieve_clip_radgraph.py \
    --img_root          "$IMG_ROOT" \
    --train_json        "$TRAIN_JSON" \
    --eval_json         "$EVAL_JSON" \
    --model_name_or_path "hf-hub:thaottn/OpenCLIP-resnet50-CC12M" \
    --checkpoint_path   "$CLIP_CHECKPOINT" \
    --output_path       "$OUTPUT_PATH" \
    --fixed_k           3 \
    --radgraph_model    radgraph-xl \
    --radgraph_batch    16
