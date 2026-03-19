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

# Stage downloads on the network volume, not the container's overlay disk
export TMPDIR="$MODELS_PATH/tmp"
mkdir -p "$TMPDIR"

echo "=== Animal Farm model download ==="
echo "Target: $MODELS_PATH"
echo ""

echo "--- Installing huggingface_hub ---"
pip install -q huggingface_hub

if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    echo "HF_TOKEN loaded from $HOME/.cache/huggingface/token"
fi
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set — downloads will be rate-limited. Set HF_TOKEN or log in with huggingface-cli."
fi

echo "--- HuggingFace: florence2 (microsoft/Florence-2-large) ---"
python3 -c "
import os
from huggingface_hub import list_repo_files, hf_hub_download
token = os.environ.get('HF_TOKEN') or None
repo_id = 'microsoft/Florence-2-large'
cache_dir = '$MODELS_PATH/huggingface'
files = sorted(list_repo_files(repo_id, token=token))
print(f'{len(files)} files to fetch')
for i, f in enumerate(files, 1):
    print(f'[{i}/{len(files)}] {f}')
    hf_hub_download(repo_id=repo_id, filename=f, cache_dir=cache_dir, token=token)
"

echo ""
echo "--- HuggingFace: moondream (vikhyatk/moondream2, rev 2025-06-21) ---"
python3 -c "
import os
from huggingface_hub import list_repo_files, hf_hub_download
token = os.environ.get('HF_TOKEN') or None
repo_id = 'vikhyatk/moondream2'
revision = '2025-06-21'
cache_dir = '$MODELS_PATH/huggingface'
files = sorted(list_repo_files(repo_id, revision=revision, token=token))
print(f'{len(files)} files to fetch')
for i, f in enumerate(files, 1):
    print(f'[{i}/{len(files)}] {f}')
    hf_hub_download(repo_id=repo_id, filename=f, revision=revision, cache_dir=cache_dir, token=token)
"

echo ""
echo "--- BLIP2: BLIP caption large (Salesforce CDN) ---"
bash "$REPO_DIR/BLIP2/download_models.sh" "$MODELS_PATH/lavis"

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
echo "--- sam3: facebook/sam3 (restricted — requires HF_TOKEN) ---"
bash "$REPO_DIR/sam3/download_models.sh" "$MODELS_PATH/sam3"

echo ""
echo "=== All models downloaded to $MODELS_PATH ==="
echo "Start the stack with:"
echo "  export MODELS_PATH=$MODELS_PATH"
echo "  docker compose up"
