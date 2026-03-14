#!/bin/bash
# Download llava-llama3 model files from HuggingFace.
# Run once before first container start. Files are volume-mounted into the container.
set -e

MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$MODELS_DIR"

hf download xtuner/llava-llama-3-8b-v1_1-gguf \
    llava-llama-3-8b-v1_1-int4.gguf \
    llava-llama-3-8b-v1_1-mmproj-f16.gguf \
    --local-dir "$MODELS_DIR"

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.gguf
