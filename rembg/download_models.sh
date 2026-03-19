#!/bin/bash
# Download rembg birefnet-general model weights.
# Run once on the RunPod volume before first container start.
# Files are volume-mounted into the container at /models/rembg (U2NET_HOME).
set -e

MODELS_DIR="${1:-$(dirname "$0")/models}"
mkdir -p "$MODELS_DIR"

pip install -q rembg onnxruntime

echo "Downloading birefnet-general model..."
U2NET_HOME="$MODELS_DIR" python3 -c "
from rembg import new_session
new_session('birefnet-general')
"

echo "Done. Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR"
