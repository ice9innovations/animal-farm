#!/bin/bash
# Download rembg model weights: birefnet-general (rembg backend) and BEN2_Base.pth (ben2 backend).
# Run once on the RunPod volume before first container start.
# Files are volume-mounted into the container at /models/rembg.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

download() {
    local url="$1"
    local dest="$2"
    echo "Connecting to $(echo "$url" | cut -d/ -f3)..."
    if [ -n "$HF_TOKEN" ]; then
        wget -c --header="Authorization: Bearer $HF_TOKEN" --connect-timeout=30 --progress=bar:force -O "$dest" "$url"
    else
        wget -c --connect-timeout=30 --progress=bar:force -O "$dest" "$url"
    fi
}

echo "Downloading birefnet-general.onnx (928MB)..."
download "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx" \
    "$MODELS_DIR/birefnet-general.onnx"

echo "Downloading BEN2_Base.pth (1.1GB)..."
download "https://huggingface.co/PramaLLC/BEN2/resolve/main/BEN2_Base.pth" \
    "$MODELS_DIR/BEN2_Base.pth"

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"
