#!/bin/bash
# Download Qwen3-VL model files from HuggingFace.
# Run once before first container start. Files are volume-mounted into the container.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

QWEN_GGUF_REPO="${QWEN_GGUF_REPO:-lmstudio-community/Qwen3-VL-2B-Instruct-GGUF}"
QWEN_MODEL_FILE="${QWEN_MODEL_FILE:-Qwen3-VL-2B-Instruct-Q4_K_M.gguf}"
QWEN_MMPROJ_FILE="${QWEN_MMPROJ_FILE:-mmproj-Qwen3-VL-2B-Instruct-F16.gguf}"
HF_BASE="https://huggingface.co/$QWEN_GGUF_REPO/resolve/main"

download() {
    local url="$1"
    local dest="$2"
    if [ -n "$HF_TOKEN" ]; then
        wget -c --header="Authorization: Bearer $HF_TOKEN" -P "$dest" "$url"
    else
        wget -c -P "$dest" "$url"
    fi
}

echo "Downloading $QWEN_MODEL_FILE..."
download "$HF_BASE/$QWEN_MODEL_FILE" "$MODELS_DIR"

echo "Downloading $QWEN_MMPROJ_FILE..."
download "$HF_BASE/$QWEN_MMPROJ_FILE" "$MODELS_DIR"

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.gguf
