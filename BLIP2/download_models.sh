#!/bin/bash
# Download BLIP caption model files from Salesforce storage.
# Run once on the RunPod network volume before starting the service.
#
# Models are stored under $TORCH_HOME/hub/checkpoints/ so timm finds them
# without re-downloading. Pass the target cache directory as the first argument,
# or set MODEL_DIR when running this script directly.
set -e

MODEL_DIR="${1:-${MODEL_DIR:-$(dirname "$0")/models}}"
TARGET="$MODEL_DIR/hub/checkpoints"
mkdir -p "$TARGET"

BASE_URL="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models"

echo "Downloading BLIP large pretrained weights..."
wget -c -q --show-progress -P "$TARGET" "$BASE_URL/model_large.pth"

echo "Downloading BLIP large caption finetuned weights..."
wget -c -q --show-progress -P "$TARGET" "$BASE_URL/model_large_caption.pth"

echo "Done. Files in $TARGET:"
ls -lh "$TARGET"
