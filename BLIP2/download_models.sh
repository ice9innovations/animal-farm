#!/bin/bash
# Download BLIP caption model files from Salesforce storage.
# Run once on the RunPod volume before first container start.
# Files are volume-mounted into the container at /export/home/.cache/lavis.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
TARGET="$MODELS_DIR/sfr-vision-language-research/BLIP/models"
mkdir -p "$TARGET"

BASE_URL="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models"

echo "Downloading BLIP large pretrained weights..."
wget -c -q --show-progress -P "$TARGET" "$BASE_URL/model_large.pth"

echo "Downloading BLIP large caption finetuned weights..."
wget -c -q --show-progress -P "$TARGET" "$BASE_URL/model_large_caption.pth"

echo "Done. Files in $TARGET:"
ls -lh "$TARGET"
