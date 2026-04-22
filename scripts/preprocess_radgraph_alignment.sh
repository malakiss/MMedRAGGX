#!/usr/bin/env bash
# Enrich the DPO alignment JSON with RadGraph-XL entities.
# Run this ONCE before DPO training.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

python ./train/medgemma/preprocess_radgraph_alignment.py \
    --input  ./data/training/alignment/radiology/radiology_report.json \
    --output ./data/training/alignment/radiology/radiology_report_radgraph.json \
    --model_type radgraph-xl \
    --batch_size 16
