#!/bin/bash
# Download Qwen2-VL-2B model files from HuggingFace.
# Run once before first container start. Files are volume-mounted into the container.
set -e

MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$MODELS_DIR"

hf download bartowski/Qwen2-VL-2B-Instruct-GGUF \
    Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
    mmproj-Qwen2-VL-2B-Instruct-f16.gguf \
    --local-dir "$MODELS_DIR"

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.gguf
