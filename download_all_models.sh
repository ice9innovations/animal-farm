#!/bin/bash
# Download all Animal Farm model weights to a target directory.
# Run once on a RunPod CPU pod before starting the GPU pod.
#
# Usage:
#   bash download_all_models.sh /workspace
#
# The target directory becomes MODELS_PATH in docker-compose:
#   export MODELS_PATH=/workspace
#   docker compose up
set -e

MODELS_PATH="${1:?Usage: $0 <models_path>  e.g. $0 /workspace}"

REPO_DIR="$(dirname "$0")"

echo "=== Animal Farm model download ==="
echo "Target: $MODELS_PATH"
echo ""

echo "--- Installing huggingface_hub ---"
pip install -q huggingface_hub

echo "--- HuggingFace: florence2 (microsoft/Florence-2-large) ---"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/Florence-2-large', cache_dir='$MODELS_PATH/huggingface')
"

echo ""
echo "--- HuggingFace: moondream (vikhyatk/moondream2, rev 2025-06-21) ---"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('vikhyatk/moondream2', revision='2025-06-21', cache_dir='$MODELS_PATH/huggingface')
"

echo ""
echo "--- LAVIS: BLIP caption large (Salesforce CDN) ---"
bash "$REPO_DIR/LAVIS/download_models.sh" "$MODELS_PATH/lavis"

echo ""
echo "--- llama-cpp: llava-llama3 GGUFs ---"
bash "$REPO_DIR/llama-cpp/download_models.sh" "$MODELS_PATH/llama-cpp"

echo ""
echo "--- qwen-cpp: Qwen GGUFs ---"
bash "$REPO_DIR/qwen-cpp/download_models.sh" "$MODELS_PATH/qwen-cpp"

echo ""
echo "--- yolov8: yolov8l.pt ---"
bash "$REPO_DIR/yolov8/download_models.sh" "$MODELS_PATH/yolov8"

echo ""
echo "--- rembg: birefnet-general ---"
bash "$REPO_DIR/rembg/download_models.sh" "$MODELS_PATH/rembg"

echo ""
echo "--- ocr: PaddleOCR det/rec/cls ---"
bash "$REPO_DIR/ocr/download_models.sh" "$MODELS_PATH/paddleocr"

echo ""
echo "=== All models downloaded to $MODELS_PATH ==="
echo "Start the stack with:"
echo "  export MODELS_PATH=$MODELS_PATH"
echo "  docker compose up"
