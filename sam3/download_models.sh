#!/bin/bash
# Download SAM3 checkpoint from HuggingFace (facebook/sam3).
# Run once before first container start. Files are volume-mounted into the container.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is required — facebook/sam3 is a restricted model."
    echo "Set HF_TOKEN before running this script."
    exit 1
fi

HF_BASE="https://huggingface.co/facebook/sam3/resolve/main"

download() {
    local url="$1"
    local dest="$2"
    if [ -n "$HF_TOKEN" ]; then
        wget -c --header="Authorization: Bearer $HF_TOKEN" -P "$dest" "$url"
    else
        wget -c -P "$dest" "$url"
    fi
}

echo "Downloading sam3.pt (3.3GB)..."
download "$HF_BASE/sam3.pt" "$MODELS_DIR"

echo "Downloading config.json..."
download "$HF_BASE/config.json" "$MODELS_DIR"

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"
