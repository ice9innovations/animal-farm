#!/bin/bash
# Download Qwen2-VL-2B model files from HuggingFace.
# Run once before first container start. Files are volume-mounted into the container.
# Source: bartowski/Qwen2-VL-2B-Instruct-GGUF (confirmed in issues/containerize-llama-server-binary.md)
# Note: docker-compose.yaml references Qwen3VL-4B filenames — reconcile before deploying.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

HF_BASE="https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main"

download() {
    local url="$1"
    local dest="$2"
    if [ -n "$HF_TOKEN" ]; then
        wget -c --header="Authorization: Bearer $HF_TOKEN" -P "$dest" "$url"
    else
        wget -c -P "$dest" "$url"
    fi
}

echo "Downloading Qwen2-VL-2B-Instruct-Q4_K_M.gguf (986MB)..."
download "$HF_BASE/Qwen2-VL-2B-Instruct-Q4_K_M.gguf" "$MODELS_DIR"

echo "Downloading mmproj-Qwen2-VL-2B-Instruct-f16.gguf (1.33GB)..."
download "$HF_BASE/mmproj-Qwen2-VL-2B-Instruct-f16.gguf" "$MODELS_DIR"

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.gguf
