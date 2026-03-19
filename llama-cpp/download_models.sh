#!/bin/bash
# Download llava-llama3 model files from HuggingFace.
# Run once before first container start. Files are volume-mounted into the container.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

HF_BASE="https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf/resolve/main"

download() {
    local url="$1"
    local dest="$2"
    if [ -n "$HF_TOKEN" ]; then
        wget -c --header="Authorization: Bearer $HF_TOKEN" -P "$dest" "$url"
    else
        wget -c -P "$dest" "$url"
    fi
}

echo "Downloading llava-llama-3-8b-v1_1-int4.gguf (4.9GB)..."
download "$HF_BASE/llava-llama-3-8b-v1_1-int4.gguf" "$MODELS_DIR"

echo "Downloading llava-llama-3-8b-v1_1-mmproj-f16.gguf (624MB)..."
download "$HF_BASE/llava-llama-3-8b-v1_1-mmproj-f16.gguf" "$MODELS_DIR"

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.gguf
