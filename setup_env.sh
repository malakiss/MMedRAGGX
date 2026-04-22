#!/usr/bin/env bash
# Create the mmedrag conda environment and install all dependencies.
#
# Usage:
#   bash setup_env.sh
#
# After setup, activate with:
#   conda activate mmedrag

set -euo pipefail

ENV_NAME="mmedrag"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Creating conda env: ${ENV_NAME} ==="
conda create -n "${ENV_NAME}" python=3.10 -y

echo "=== Installing pip packages ==="
conda run -n "${ENV_NAME}" pip install \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r "${REPO_DIR}/requirements_medgemma.txt"

echo "=== Installing OpenCLIP (editable) ==="
conda run -n "${ENV_NAME}" pip install -e "${REPO_DIR}/train/open_clip"

echo ""
echo "=== Setup complete! ==="
echo "Activate with:  conda activate ${ENV_NAME}"
echo ""
echo "Required environment variables:"
echo "  export HF_TOKEN=<your huggingface token>"
echo "  export WANDB_API_KEY=<your wandb key>"
