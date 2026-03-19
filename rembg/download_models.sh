#!/bin/bash
# Download rembg model weights: birefnet-general (rembg backend) and BEN2_Base.pth (ben2 backend).
# Run once on the RunPod volume before first container start.
# Files are volume-mounted into the container at /models/rembg.
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

export TMPDIR="${TMPDIR:-$MODELS_DIR}"
pip install -q rembg onnxruntime

echo "Downloading birefnet-general model..."
U2NET_HOME="$MODELS_DIR" python3 -c "
from rembg import new_session
new_session('birefnet-general')
"

echo "Connecting to HuggingFace for BEN2_Base.pth (420MB)..."
if [ -n "$HF_TOKEN" ]; then
    wget -c --header="Authorization: Bearer $HF_TOKEN" --connect-timeout=30 --progress=bar:force \
        -P "$MODELS_DIR" "https://huggingface.co/PramaLLC/BEN2/resolve/main/BEN2_Base.pth"
else
    wget -c --connect-timeout=30 --progress=bar:force \
        -P "$MODELS_DIR" "https://huggingface.co/PramaLLC/BEN2/resolve/main/BEN2_Base.pth"
fi

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"
